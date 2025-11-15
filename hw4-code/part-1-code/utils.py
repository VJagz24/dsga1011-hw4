import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Synonym replacement transformation
    # This replaces random words with their synonyms using WordNet
    
    text = example["text"]
    words = word_tokenize(text)
    
    # Probability of replacing each word
    replace_prob = 0.35  # 35% of words will be replaced
    
    new_words = []
    for word in words:
        # Randomly decide whether to replace this word
        if random.random() < replace_prob:
            # Get synsets for the word
            synsets = wordnet.synsets(word)
            
            if synsets:
                # Get all lemmas (synonyms) from all synsets
                synonyms = []
                for synset in synsets:
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        # Only use single-word synonyms and avoid the original word
                        if ' ' not in synonym and synonym.lower() != word.lower():
                            synonyms.append(synonym)
                
                # If we found synonyms, randomly pick one
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    
    # Detokenize back to text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example