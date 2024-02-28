'''
Concrete IO class for a specific dataset
'''
import torch

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src_code.base_class.dataset import dataset
import re
import numpy as np
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-1 * x))

    def tokenize_sentence(self,sentence):
        # Split by whitespace and punctuation
        sentence = sentence.lower()
        tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
        return tokens

    def jokesProcess(self):
        jokes = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip()
            line = line.split(",")
            line = line[1]
            tokenized_joke = self.tokenize_sentence(line.strip())
            jokes.append(tokenized_joke)
        jokes = jokes[1:]
        return jokes

    def vocabSet(self,jokes):
        vocab = set()
        for joke in jokes:
            vocab.update(joke)
        vocab_size = len(vocab)
        vocab = {word: index / vocab_size for index, word in enumerate(sorted(vocab))}
        print("Length of vocab",len(vocab))
        return vocab

    def encode(self, jokes, vocab):
        encoded_jokes = []
        for joke in jokes:
            #print(joke)
            encoded_joke = [vocab[token] for token in joke]
            encoded_jokes.append(encoded_joke)
        return encoded_jokes

    def decode(self, encoded_jokes, vocab):
        decoded_jokes = []

        for encoded_joke in encoded_jokes:
            for key,val in vocab.items():
                if val == encoded_joke:
                    decoded_jokes.append(key)
        return decoded_jokes

    def decodeTest(self, encoded_joke, vocab):
        decode = []
        for token in encoded_joke:
            sortedVocab = sorted(vocab.keys())
            token = token * len(vocab)
            curr = round(token)
            decode.append(sortedVocab[curr])
        return decode

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        jokes = self.jokesProcess()
        vocab = self.vocabSet(jokes)
        encoded_jokes = self.encode(jokes, vocab)
        decoded_jokes = self.decode(encoded_jokes[0],vocab)

        input = []
        output = []
        for seq in encoded_jokes:
            #for i in range(3, len(seq)):
            input.append(seq[0:3])
            output.append(seq[3:])
        max_output_length = max(len(seq) for seq in output)
        padded_output = [seq + [0] * (max_output_length - len(seq)) for seq in output]
        padded_input = [seq + [0] * (max_output_length - len(seq)) for seq in input]
        input = np.array(padded_input)
        output = np.array(padded_output)

        print("Vocabulary:", vocab)
        print("Encoded Jokes:", encoded_jokes[0])
        print("Decoded Jokes:", decoded_jokes)
        print("Decode length", len(decoded_jokes))
        print("encode length", len(encoded_jokes[0]))
        f.close()
        return input, output

curr = Dataset_Loader()
curr.dataset_source_folder_path = "/Users/Srihita/Desktop/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/"
curr.dataset_source_file_name = "data"
input, output = curr.load()
jokes = curr.jokesProcess()
vocab = curr.vocabSet(jokes)
