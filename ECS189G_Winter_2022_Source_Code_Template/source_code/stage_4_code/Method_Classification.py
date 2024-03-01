from sklearn.metrics import classification_report
from tqdm import tqdm

from source_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy


# import numpy as np
# from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
#
#
# import pandas as pd
# import numpy as np
# from keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
# from keras.layers.embeddings import Embedding
# from keras.models import Model
# import string
# import re
# from keras.preprocessing.text import Tokenizer
# from sklearn.preprocessing import LabelBinarizer
# from keras.preprocessing.sequence import pad_sequences
# import keras
# from sklearn.model_selection import train_test_split


class Method_Classification(method, nn.Module):

    input_size = 1 # have to change
    hidden_size = 256
    num_layers = 2
    num_classes = 1  # Positive or Negative sentiment
    sequence_length = 912  # Number of embeddings to process in sequence
    batch_size = 20
    embedding_size = 100
    learning_rate = 0.001
    max_epochs = 10
    grad_clip = 5
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': max_epochs
    }

    losses = []
    accuracies = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        super(Method_Classification, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.input_size = 100
        self.vocab_size = 69966

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)  # Fully connected layer
        self.sig = nn.Sigmoid()

    def initialize_embeddings(self, embedding_matrix):
        """Initialize the embedding layer with preloaded weights."""
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        x = x.long()

        x_embedded = self.embedding(x)
        # print("embedding", x_embedded.shape ,x_embedded)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hidden = self.lstm(x_embedded)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        # print("out", out)
        # print("hidden", hidden.squeeze(0))
        out = self.sig(out)
        return out

    def train_model(self, train_loader):
        self.train()  # Set the model to training mode
        # epochloop = tqdm(range(self.max_epochs), position=0, desc='Training', leave=True)

        # early stop trigger
        es_trigger = 0
        val_loss_min = torch.inf
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        print("train_loader size (in train_model)", len(train_loader))
        for epoch in range(self.max_epochs):

            train_loss = 0
            train_acc = 0
            total_loss = 0
            print(f'\nEpoch {epoch + 1}/{self.max_epochs}')
            for id, (feature, target) in enumerate(train_loader):
                # Forward pass
                # print("Shape of data: train", data.shape)
                # epochloop.set_postfix_str(f'Training batch {id}/{len(train_loader)}')

                self.optimizer.zero_grad()

                out = self.forward(feature)

                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5])
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                train_acc += acc.item()

                loss = self.criterion(out.squeeze(), target.float())
                train_loss += loss.item()
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()

                del feature, target, predicted

                # self.optimizer.zero_grad()
                # outputs = self.forward(data)
                # loss = self.criterion(outputs.squeeze(), targets.float())
                # # self.losses.append(loss.item())
                #
                # # Backward and optimize
                # loss.backward()
                # self.optimizer.step()
            print("Avg loss", train_loss / len(train_loader))
            print("Accuracy", train_acc / len(train_loader))
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['train_acc'].append(train_acc / len(train_loader))

            # total_loss += loss.item()
            # avg_loss = total_loss / len(train_loader)
            # print(f'Average Loss: {avg_loss}')
            # self.losses.append(total_loss)
    def test_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        test_loss = 0
        test_acc = 0

        all_target = []
        all_predicted = []

        testloop = tqdm(test_loader, leave=True, desc='Inference')

        with torch.no_grad():
            for feature, target in testloop:

                out = self.forward(feature)

                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5])
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                test_acc += acc.item()

                loss = self.criterion(out.squeeze(), target.float())
                test_loss += loss.item()

                all_target.extend(target.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
            print(f'Accuracy: {test_acc / len(test_loader):.4f}, Loss: {test_loss / len(test_loader):.4f}')
        print(classification_report(all_predicted, all_target))

        # total = 0
        # correct = 0
        # all_preds = []
        # all_targets = []
        # with torch.no_grad():
        #     for data, targets in test_loader:
        #         # print("Shape of data: test", data.shape)
        #         outputs = self.forward(data)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += targets.size(0)
        #         correct += (predicted == targets).sum().item()
        #         all_preds.append(predicted)
        #         all_targets.append(targets)
        #
        # all_preds = torch.cat(all_preds)
        # all_targets = torch.cat(all_targets)
        #
        # all_preds = all_preds.numpy()
        # all_targets = all_targets.numpy()
        #
        # accuracy = 100 * correct / total
        # print(f'Accuracy: {accuracy}%')
        # self.accuracies.append(accuracy)
        # return {'pred_y': all_preds, 'true_y': all_targets }

    def run(self):

        print("size of train:", self.data['train']['X'].shape, self.data['train']['y'].shape)
        print("size of test:", self.data['test']['X'].shape, self.data['test']['y'].shape)

        print("size of each entry", self.data['train']['X'][0].shape)
        print("check", self.data['train']['X'][0], self.data['train']['y'][0])

        self.initialize_embeddings(self.data['embedding'])
        print("initialized embedding to glove embedding")

        train_dataset = TensorDataset(self.data['train']['X'], self.data['train']['y'])
        test_dataset = TensorDataset(self.data['test']['X'], self.data['test']['y'])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # for epoch in range(self.max_epochs):
        #     print(f'Epoch {epoch + 1}/{self.max_epochs}')
        self.train_model(train_loader)
        test_results = self.test_model(test_loader)

        self.plot_loss()
        self.plot_accuracy()

        print("pred_y size", len(np.array(test_results['pred_y'])))
        print("true_y size", len(test_results['true_y']))

        return {'pred_y': test_results['pred_y'], 'true_y': test_results['true_y']}

    def plot_loss(self):
        plt.plot(self.history['train_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_4_result/classification_loss.png')

    def plot_accuracy(self):
        plt.plot(self.history['train_acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.savefig('../../result/stage_4_result/classification_accuracy.png')


    # def read_glove_vector(self, glove_vec):
    #     with open(glove_vec, 'r', encoding='UTF-8') as f:
    #         words = set()
    #         word_to_vec_map = {}
    #         for line in f:
    #             w_line = line.split()
    #             curr_word = w_line[0]
    #             word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    #
    #     return word_to_vec_map
    #
    # def imdb_rating(self, input_shape):
    #
    #     X_indices = Input(input_shape)
    #
    #     embeddings = self.embedding_layer(X_indices)
    #
    #     X = LSTM(128, return_sequences=True)(embeddings)
    #
    #     X = Dropout(0.6)(X)
    #
    #     X = LSTM(128, return_sequences=True)(X)
    #
    #     X = Dropout(0.6)(X)
    #
    #     X = LSTM(128)(X)
    #
    #     X = Dense(1, activation='sigmoid')(X)
    #
    #     model = Model(inputs=X_indices, outputs=X)
    #
    #     return model
    #
    # def conv1d_model(self, input_shape):
    #
    #     X_indices = Input(input_shape)
    #
    #     embeddings = self.embedding_layer(X_indices)
    #
    #     X = Conv1D(512, 3, activation='relu')(embeddings)
    #
    #     X = MaxPooling1D(3)(X)
    #
    #     X = Conv1D(256, 3, activation='relu')(X)
    #
    #     X = MaxPooling1D(3)(X)
    #
    #     X = Conv1D(256, 3, activation='relu')(X)
    #     X = Dropout(0.8)(X)
    #     X = MaxPooling1D(3)(X)
    #
    #     X = GlobalMaxPooling1D()(X)
    #
    #     X = Dense(256, activation='relu')(X)
    #     X = Dense(1, activation='sigmoid')(X)
    #
    #     model = Model(inputs=X_indices, outputs=X)
    #
    #     return model
    # def run(self):
    #     print('method running...')
    #     print('--start training...')
    #
    #     tokenizer = Tokenizer(num_words=5000)
    #     tokenizer.fit_on_texts(self.data['train']['X'])
    #
    #     words_to_index = tokenizer.word_index
    #
    #     word_to_vec_map = self.read_glove_vector('../../stage_4_data/glove.6B/glove.6B.50d.txt')
    #
    #     maxLen = 150
    #
    #     vocab_len = len(words_to_index)
    #     embed_vector_len = word_to_vec_map['moon'].shape[0]
    #
    #     emb_matrix = np.zeros((vocab_len, embed_vector_len))
    #
    #     for word, index in words_to_index.items():
    #         embedding_vector = word_to_vec_map.get(word)
    #         if embedding_vector is not None:
    #             emb_matrix[index, :] = embedding_vector
    #
    #     self.embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen,
    #                                 weights=[emb_matrix], trainable=False)
    #     model = self.imdb_rating((maxLen,))
    #     model.summary()
    #
    #     model_1d = self.conv1d_model((maxLen,))
    #     model_1d.summary()
    #
    #     X_train_indices = tokenizer.texts_to_sequences(self.data['train']['X'])
    #     X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
    #     # X_train_indices.shape
    #
    #     adam = keras.optimizers.Adam(learning_rate=0.0001)
    #     model_1d.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    #
    #     model_1d.fit(X_train_indices, self.data['train']['y'], batch_size=64, epochs=15)
    #
    #     X_test_indices = tokenizer.texts_to_sequences(self.data['test']['X'])
    #
    #     X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')
    #
    #     model.evaluate(X_test_indices, self.data['test']['y'])
    #
    #     model_1d.evaluate(X_test_indices, self.data['test']['y'])
    #
    #     preds = model_1d.predict(X_test_indices)
    #
    #     n = np.random.randint(0, 9999)
    #     print(self.data['test']['X'][n])
    #
    #     if preds[n] > 0.5:
    #         print('predicted sentiment : positive')
    #     else:
    #         print('precicted sentiment : negative')
    #
    #     if (self.data['test']['y'][n] == 1):
    #         print('correct sentiment : positive')
    #     else:
    #         print('correct sentiment : negative')
    #
    #     model_1d.save_weights('../../stage_4_result/model_1d_weights.hdf5')
    #
    #     # reviews_list_idx = tokenizer.texts_to_sequences(reviews_list)
    #
