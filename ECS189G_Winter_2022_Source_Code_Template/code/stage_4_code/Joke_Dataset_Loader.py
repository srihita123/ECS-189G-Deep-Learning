'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import csv

class Joke_Dataset_Loader(dataset):
    data = None
    vocab = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def tokenize(self, line):
        return line.split()

    def create_vocab(self):
        """Create a vocabulary from the dataset."""
        vocab = set()
        for line in self.data:
            vocab.update(line)
        vocab.update("END_OF_JOKE")
        vocab = list(vocab)
        vocab.sort()  # Sort the vocabulary for consistency
        self.vocab = vocab
        return vocab

    def encode(self, word):
        """One-hot encode a word based on the vocabulary."""
        encoding = np.zeros(len(self.vocab))  # Initialize an array of zeros
        encoding[self.vocab.index(word)] = 1  # Set the corresponding index to 1
        return encoding

    def get_encoded(self):
        return [([self.encode(token) for token in line]) for line in self.data]

    def get_encoded_single(self):
        return [self.encode(token) for token in self.data]

    def decode_from_encoding(self, encoding):
        return self.vocab[np.argmax(encoding)]

    def decode_from_index(self, index):
        return self.vocab[index]

    def load(self):
        print('loading data...')
        # set data attribute to data from file
        self.data = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            tokens = self.tokenize(line[1])
            tokens.append("END_OF_JOKE")
            self.data.append(tokens)
        f.close()
        # set vocab attribute using data
        self.create_vocab()
        # Returns list of jokes from the document where each joke is list of encoded tokens
        # And returns list of document vocabulary
        return {'doc': self.get_encoded(), 'vocab': self.vocab}

'''
if 1:
    jokes = Joke_Dataset_Loader("dataset", "dataset of jokes")
    jokes.dataset_source_folder_path = "../../data/stage_4_data/text_generation/"
    jokes.dataset_source_file_name = "data"
    loaded_jokes = jokes.load()
    print(jokes.data[9])
    print(loaded_jokes['doc'][9])
    print(loaded_jokes['vocab'])
'''
