import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        '''
        Dataset class for T5 text-to-SQL task.
        
        Args:
            data_folder: Path to data directory
            split: 'train', 'dev', or 'test'
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Load data
        self.data = self.process_data(data_folder, split, self.tokenizer)
    
    def process_data(self, data_folder, split, tokenizer):
        '''
        Process the natural language and SQL query data.
        
        Returns:
            List of tuples (encoder_ids, decoder_ids) for train/dev
            List of encoder_ids for test
        '''
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        # For train and dev, also load SQL queries
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
            
            data = []
            for nl, sql in zip(nl_queries, sql_queries):
                # Tokenize natural language input
                # Add prefix for task specification
                encoder_input = f"translate English to SQL: {nl}"
                encoder_ids = tokenizer(
                    encoder_input,
                    add_special_tokens=True,
                    return_tensors='pt'
                ).input_ids.squeeze(0)
                
                # Tokenize SQL output
                decoder_ids = tokenizer(
                    sql,
                    add_special_tokens=True,
                    return_tensors='pt'
                ).input_ids.squeeze(0)
                
                data.append((encoder_ids, decoder_ids))
        else:
            # For test set, only process natural language queries
            data = []
            for nl in nl_queries:
                encoder_input = f"translate English to SQL: {nl}"
                encoder_ids = tokenizer(
                    encoder_input,
                    add_special_tokens=True,
                    return_tensors='pt'
                ).input_ids.squeeze(0)
                
                data.append(encoder_ids)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Collation function for training and evaluation (dev set).
    
    Args:
        batch: List of (encoder_ids, decoder_ids) tuples
    
    Returns:
        encoder_ids: Padded encoder inputs [B, T]
        encoder_mask: Attention mask for encoder [B, T]
        decoder_inputs: Decoder inputs (shifted right) [B, T']
        decoder_targets: Target tokens for decoder [B, T']
        initial_decoder_inputs: First decoder token [B, 1]
    '''
    encoder_ids_list = []
    decoder_ids_list = []
    
    for encoder_ids, decoder_ids in batch:
        encoder_ids_list.append(encoder_ids)
        decoder_ids_list.append(decoder_ids)
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Prepare decoder inputs and targets
    # T5 uses pad token (0) as the start token for decoder
    decoder_inputs_list = []
    decoder_targets_list = []
    initial_decoder_inputs_list = []
    
    for decoder_ids in decoder_ids_list:
        # Decoder input: shift right by prepending pad token
        decoder_input = torch.cat([torch.tensor([PAD_IDX]), decoder_ids[:-1]])
        decoder_inputs_list.append(decoder_input)
        
        # Decoder target: original sequence
        decoder_targets_list.append(decoder_ids)
        
        # Initial decoder input (for generation)
        initial_decoder_inputs_list.append(torch.tensor([PAD_IDX]))
    
    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs_list).unsqueeze(1)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function for test set (no targets available).
    
    Args:
        batch: List of encoder_ids tensors
    
    Returns:
        encoder_ids: Padded encoder inputs [B, T]
        encoder_mask: Attention mask for encoder [B, T]
        initial_decoder_inputs: First decoder token [B, 1]
    '''
    encoder_ids_list = batch
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input (pad token for T5)
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader