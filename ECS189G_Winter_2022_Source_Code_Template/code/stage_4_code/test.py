import torch
from RNN import model
from src_code.stage_4_code.Dataset_Loader import Dataset_Loader
# Assuming you have already trained your model and have it stored in a variable called 'model'
# Also assuming you have a dictionary index_word mapping indices to words

# Convert the input words to indices using your index_word dictionary
curr = Dataset_Loader()
curr.dataset_source_folder_path = "/Users/Srihita/Desktop/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/"
curr.dataset_source_file_name = "data"
jokes = curr.jokesProcess()
vocab = curr.vocabSet(jokes)

word_index = {word: index for index, word in vocab.items()}
input_words = ["what", "did", "the"]
input_indices = [word_index[word] for word in input_words]

# Convert input_indices to a PyTorch tensor
input_tensor = torch.LongTensor(input_indices).unsqueeze(0)  # Add batch dimension

# Set model to evaluation mode
model.eval()

# Initialize hidden state
hidden = model.init_hidden(1)  # Assuming batch size is 1

# Forward pass
with torch.no_grad():
    output, _ = model(input_tensor, hidden)

# Get the predicted indices
_, predicted_indices = torch.max(output, dim=2)

# Convert predicted indices to words
predicted_words = [vocab[idx.item()] for idx in predicted_indices.squeeze()]

# Print the continuation of the joke
print(" ".join(predicted_words))
