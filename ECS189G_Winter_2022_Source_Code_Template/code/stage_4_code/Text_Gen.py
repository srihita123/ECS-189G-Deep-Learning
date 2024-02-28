import numpy as np


def create_vocab(data):
    """Create a vocabulary from the dataset."""
    vocab = set()
    for line in data:
        tokens = line.split()
        vocab.update(tokens)
    vocab = list(vocab)
    vocab.sort()  # Sort the vocabulary for consistency
    return vocab


def encode(word, vocab):
    """One-hot encode a word based on the vocabulary."""
    # This could be slow because iterates over vocab list
    encoding = np.zeros(len(vocab))  # Initialize an array of zeros
    if word in vocab:
        index = vocab.index(word)  # Find the index of the word in the vocabulary
        encoding[index] = 1  # Set the corresponding index to 1
    return encoding


def encode_joke(joke, vocab):
    """One-hot encode each word in joke"""
    words = joke.split()  # Split the joke into words
    return [encode(w, vocab) for w in words]  # Encode each word


def decode(encoding, vocab):
    return vocab[np.argmax(encoding)]


def decode_joke(encoded_joke, vocab):
    """Decode the encoded joke back to a string."""
    decoded_joke = []
    for e in encoded_joke:
        decoded_joke.append(decode(e, vocab))
    return ' '.join(decoded_joke)


if 1:
    # Example dataset of jokes
    dataset = [
        "Why don't scientists trust atoms? Because they make up everything.",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised."
    ]

    # Create vocabulary
    vocab = create_vocab(dataset)

    joke = "she don't have much"
    # One-hot encode a joke
    encoded_joke = encode_joke(joke, vocab)

    decoded_joke = decode_joke(encoded_joke, vocab)
    # Print the results
    print("Joke:", joke)
    print("Encoded Joke:")
    for word, encoding in zip(joke.split(), encoded_joke):
        print(word, ":", encoding)
    print("Decoded: ", decoded_joke)
