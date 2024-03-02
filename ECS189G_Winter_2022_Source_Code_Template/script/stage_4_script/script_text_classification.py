from transformers import BertTokenizer, BertModel
import torch
import json
import pickle


def get_batches(reviews, batch_size=16):
    """Yield batches of reviews."""
    for i in range(0, len(reviews), batch_size):
        yield reviews[i:i + batch_size]


def encode_reviews(reviews, batch_size=16):
    model.eval()  # Set the model to evaluation mode
    embeddings = []
    for i in range(0, len(reviews), batch_size):
        batch_reviews = [" ".join(review) for review in reviews[i:i + batch_size]]  # Join words into strings
        # print(batch_reviews[0])  # Print the first entry to check their format
        tokens = tokenizer(batch_reviews, return_tensors='pt', padding=True, truncation=True, max_length=512)
        print(i)
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings.extend(outputs.last_hidden_state[:, 0, :].numpy())
    print("embeddings done...")
    return embeddings


def tokens_to_text(token_lists):
    """Convert lists of tokens back into strings."""
    return [" ".join(tokens) for tokens in token_lists]


if 1:
    # logging.set_verbosity_error()

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    file_path_train = 'clean_reviews_train.json'
    file_path_test = 'clean_reviews_test.json'
    file_paths = [file_path_test, file_path_train]

    output_file_path_train = "./embeddings_small_train.pkl"
    json_output_file_path_train = "./embeddings_small_train.json"
    output_file_path_test = "./embeddings_small_test.pkl"
    json_output_file_path_test = "./embeddings_small_test.json"

    for file_path in file_paths:

        with open(file_path, 'r') as file:
            cleaned_reviews = json.load(file)

        reviews_pos = tokens_to_text(cleaned_reviews['pos'])
        reviews_neg = tokens_to_text(cleaned_reviews['neg'])

        # Example: Tokenizing and encoding a single review
        # Assuming 'cleaned_review' is a single cleaned review text
        # tokens = tokenizer(cleaned_reviews, return_tensors='pt', padding=True, truncation=True, max_length=512)
        print("positive embeddings")
        positive_embeddings = encode_reviews(cleaned_reviews['pos'], batch_size=16)
        print("negative embeddings")
        negative_embeddings = encode_reviews(cleaned_reviews['neg'], batch_size=16)

        print(positive_embeddings)

        # Save both embeddings in a dictionary
        embeddings = {'positive': positive_embeddings, 'negative': negative_embeddings}
        json_embeddings = {'positive': positive_embeddings, 'negative': negative_embeddings}

        print("file path", file_path)
        if file_path == file_path_train:
            output_file_path = output_file_path_train
            json_output_file_path = json_output_file_path_train
        else:
            output_file_path = output_file_path_test
            json_output_file_path = json_output_file_path_test

        with open(output_file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        f.close()

        with open(json_output_file_path, 'w') as f:
            json.dump(json_embeddings, f, indent=4)
        f.close()
