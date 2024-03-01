# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from collections import Counter

from tqdm import tqdm

from source_code.base_class.dataset import dataset
import pickle
import os
import numpy as np
import json
import torch
# from keras.preprocessing.sequence import pad_sequences


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        print("loading from ", self.dataset_source_folder_path + self.dataset_source_file_name)
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = json.load(f)
        f.close()

        glove_embeddings = load_glove_embeddings('../../data/stage_4_data/glove.6B/glove.6B.100d.txt')
        glove_embed_size = 100

        vocab_set = set()
        for sentiment in ['pos', 'neg']:
            for review in data['train'][sentiment]:
                vocab_set.update(review)
            for review in data['test'][sentiment]:
                vocab_set.update(review)

        counter = Counter(vocab_set)
        vocab = sorted(counter, key=counter.get, reverse=True)
        int2word = dict(enumerate(vocab, 1))
        int2word[0] = '<PAD>'
        word2int = {word: id for id, word in int2word.items()}

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for sentiment in ['pos', 'neg']:
            for review in data['train'][sentiment]:
                review_enc = [word2int[word] for word in review]
                X_train.append(review_enc)
                if sentiment == 'pos':
                    y_train.append(1)
                else:
                    y_train.append(0)
            for review in data['test'][sentiment]:
                review_enc = [word2int[word] for word in review]
                X_test.append(review_enc)
                if sentiment == 'pos':
                    y_test.append(1)
                else:
                    y_test.append(0)

        for i in range(5):
            print(X_train[i][:5])

        seq_length = 256
        X_train_pad = pad_features(X_train, pad_id=word2int['<PAD>'], seq_length=seq_length)
        X_test_pad = pad_features(X_test, pad_id=word2int['<PAD>'], seq_length=seq_length)
        # features = pad_features(reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)

        assert len(X_train_pad) == len(X_train)
        assert len(X_train_pad[0]) == seq_length

        # word_to_index = {word: idx for idx, word in enumerate(vocab_set)}
        # word_to_index['<UNK>'] = 0  # Add a token for unknown words

        embedding_matrix = torch.zeros((len(vocab_set) + 1, glove_embed_size))

        for word, i in word2int.items():
            if word in glove_embeddings:
                embedding_matrix[i] = glove_embeddings[word]
            else:
                embedding_matrix[i] = 0


        print("size of vocab", len(vocab_set))
        print("size of glove", len(embedding_matrix))
        #
        # X_train, y_train = encode_and_pad(data['train'], word_to_index, glove_embeddings)
        # X_test, y_test = encode_and_pad(data['test'], word_to_index, glove_embeddings)

        return {
            'X_train':  torch.tensor(X_train_pad, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'X_test':  torch.tensor(X_test_pad, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.long),
            'embedding': embedding_matrix
        }

def pad_features(reviews, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features

def load_glove_embeddings(glove_file):
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            embeddings_dict[word] = vector
    return embeddings_dict

def custom_pad_sequences_py(sequences, maxlen, value=0):
    padded_sequences = []
    for seq in sequences:
        padded_seq = None
        if len(seq) < maxlen:
            # Calculate the number of padding elements needed
            num_padding = maxlen - len(seq)
            # Pad the sequence based on the specified padding
            padded_seq = seq + [value] * num_padding
        else:
            padded_seq = seq[:maxlen]
        padded_sequences.append(padded_seq)
    return padded_sequences


def encode_and_pad(dataset_partition, word_to_index, glove_embeddings):
    encoded_reviews = []
    labels = []
    max_length = 0

    for sentiment, reviews in dataset_partition.items():
        for review in reviews:
            # Encoding
            # for word in review:
            #     embedded_word = glove_embeddings.get(word, word_to_index['<UNK>'])
            encoded_review = [word_to_index.get(word, word_to_index['<UNK>']) for word in review]
            encoded_reviews.append(encoded_review)
            labels.append(1 if sentiment == 'pos' else 0)

            if review == reviews[0]:
                print("review size", len(review))
                print("encoded review size:", len(encoded_review))
                # for word in review:
                #     print(glove_embeddings.get(word, word_to_index['<UNK>']).shape)

            # Update max_length if this is training data
            if len(encoded_review) > max_length:
                max_length = len(encoded_review)

    # Pad sequences
    print("max length", max_length)
    padded_reviews = custom_pad_sequences_py(encoded_reviews, maxlen=max_length, value=0)

    return np.array(padded_reviews), np.array(labels)
