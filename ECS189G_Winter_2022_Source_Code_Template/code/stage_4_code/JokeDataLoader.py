'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import random
from collections import Counter
from code.base_class.dataset import dataset
from nltk.tokenize import word_tokenize

class JokeDataLoader(dataset):
    data = None
    encoded_data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_to_int = None
    int_to_vocab = None
    vocab_size = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def set_vocab(self, all_data):
        word_count = Counter(all_data)
        sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
        self.int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        self.vocab_to_int = {word: ii for ii, word in self.int_to_vocab.items()}
        self.vocab_size = len(self.int_to_vocab)

    def preprocess_jokes(self, raw_data):
        tokenized_data = []
        for joke in raw_data:
            tokens = word_tokenize(joke.lower())
            # Important to add end of joke token before shuffle and creating vocab
            tokens.append("END_OF_JOKE")
            tokenized_data.append(tokens)

        # shuffle jokes, then flatten into one list
        random.seed(42)
        random.shuffle(tokenized_data)
        tokenized_data = [token for joke in tokenized_data for token in joke]
        self.data = tokenized_data
        # set index lookup tables for tokens in dataset vocabulary
        self.set_vocab(tokenized_data)
        # encode using lookup table
        self.encoded_data = [self.vocab_to_int[token] for token in tokenized_data]

    # For encoding a single string input (for testing)
    def encode_text(self, text):
        tokens = word_tokenize(text.lower())
        return [self.vocab_to_int[token] for token in tokens]

    def decode_text(self, encodings):
        return [self.int_to_vocab[encoding] for encoding in encodings]

    def load(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        reader = csv.reader(f)
        next(reader)
        # pass Jokes column for preprocessing
        self.preprocess_jokes([line[1] for line in reader])
        f.close()
        print("loaded data!")
        return self.encoded_data

'''
test = JokeDataLoader()
test.dataset_source_folder_path = "../../data/stage_4_data/text_generation/"
test.dataset_source_file_name = "data"
test.load()
print(test.data)
print(test.encoded_data)
print(test.int_to_vocab)
print(test.vocab_to_int)
'''


