from collections import Counter
import os
import pickle
import csv
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

SPECIAL_WORDS = {'PADDING': '<PAD>'}


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    word_count = Counter(text)
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    token = dict()
    token['.'] = '<PERIOD>'
    token[','] = '<COMMA>'
    token['"'] = 'QUOTATION_MARK'
    token[';'] = 'SEMICOLON'
    token['!'] = 'EXCLAIMATION_MARK'
    token['?'] = 'QUESTION_MARK'
    token['('] = 'LEFT_PAREN'
    token[')'] = 'RIGHT_PAREN'
    token['-'] = 'QUESTION_MARK'
    token['\n'] = 'NEW_LINE'
    return token


def load_data(path):
    """
    Load Dataset from File
    """
    f = open(path, 'r')
    lines = csv.reader(f)
    # Skip header
    next(lines)
    lines = [line[1] for line in lines]
    text = "\n".join(lines)
    return text


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    print(text)
    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    n_batches = len(words) // batch_size
    x, y = [], []
    words = words[:n_batches * batch_size]

    for ii in range(0, len(words) - sequence_length):
        i_end = ii + sequence_length
        batch_x = words[ii:ii + sequence_length]
        x.append(batch_x)
        batch_y = words[i_end]
        y.append(batch_y)

    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    # return a dataloader
    return data_loader


class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function

        # define embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # define lstm layer
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # define model layers
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function
        batch_size = x.size(0)
        x = x.long()

        # embedding and lstm_out
        embeds = self.embedding(x)
        rnn_out, hidden = self.rnn(embeds, hidden)

        # stack up lstm layers
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)

        # dropout, fc layer and final sigmoid layer
        out = self.fc(rnn_out)

        # reshaping out layer to batch_size * seq_length * output_size
        out = out.view(batch_size, -1, self.output_size)

        # return last batch
        out = out[:, -1]

        # return one batch of output word scores and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weights.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        # initialize hidden state with zero weights, and move to GPU if available

        return hidden


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    # TODO: Implement Function

    # creating variables for hidden state to prevent back-propagation
    # of historical states
    h = tuple([each.data for each in hidden])

    rnn.zero_grad()
    # move inputs, targets to GPU
    '''
    inputs, targets = inp.cuda(), target.cuda()
    '''
    inputs, targets = inp, target

    output, h = rnn(inputs, h)

    loss = criterion(output, targets)

    # perform backpropagation and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []

    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if (batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))

        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probabilities
        p = F.softmax(output, dim=1).data

        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())

        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')

    # return all the sentences
    return gen_sentences


''' ------------------RUNS THIS CODE BELOW ----------------------------'''

# Process data
preprocess_and_save_data('../../data/stage_4_data/text_generation/data', token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

if 0:
    # Data params
    # Sequence Length
    sequence_length = 10  # of words in a sequence
    # Batch Size
    batch_size = 128

    # data loader - do not change
    train_loader = batch_data(int_text, sequence_length, batch_size)

    # Training parameters
    # Number of Epochs
    num_epochs = 10
    # Learning Rate
    learning_rate = 0.001

    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = vocab_size
    # Embedding Dimension
    embedding_dim = 200
    # Hidden Dimension
    hidden_dim = 250
    # Number of RNN Layers
    n_layers = 2

    # Show stats for every n number of batches
    show_every_n_batches = 100

    # create model and move to gpu if available
    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training the model
    trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

    # saving the trained model
    save_model('./save/trained_rnn', trained_rnn)
    print('Model Trained and Saved')
if 1:
    gen_length = 20  # modify the length to your preference
    prime_words = ['how']  # name for starting the script
    sequence_length = 10  # of words in a sequence

    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    trained_rnn = load_model('./save/trained_rnn')
    for prime_word in prime_words:
        pad_word = SPECIAL_WORDS['PADDING']
        generated_script = generate(trained_rnn, vocab_to_int[prime_word], int_to_vocab, token_dict,
                                    vocab_to_int[pad_word], gen_length)
        print(generated_script)