import json
import os
from nltk import sent_tokenize
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# nltk.download()
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


class Clean_Reviews:
    def __init__(self, folder):
        self.folder = folder
        self.processed_reviews = {
            'pos': [],
            'neg': []
        }

    def clean_text(self, text):
        # Tokenize the text
        sentences = sent_tokenize(text)
        all_tokens = []

        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())

            # Remove stopwords
            remove_words = set(stopwords.words('english'))
            remove_words.add("br")
            tokens = [word for word in tokens if word not in remove_words]

            punc = string.punctuation

            # Characters to remove
            chars_to_remove = "'-"

            # Remove specified characters from the punctuation string
            punc_filtered = ''.join([char for char in punc if char not in chars_to_remove])

            # print(punc_filtered)

            table = str.maketrans('', '', punc_filtered)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in words]

            all_tokens.extend(tokens)


        return all_tokens

    def load_reviews(self, label):
        reviews = []
        subfolder = 'pos' if label == 'pos' else 'neg'
        # folder_path = os.path.join(self.folder, subfolder)
        folder_path = self.folder + '/' + subfolder
        # folder_path = '/'
        print("folder path", folder_path)
        # count = 0
        for filename in os.listdir(folder_path):
            # print(filename)
            # count += 1
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                review = file.read()
                reviews.append(review)
            # if count == 100:
            #     break
            # file = open(filename, 'rt')
            # review = file.read()
            # reviews.append(review)
            # file.close()


        return reviews

    def preprocess_reviews(self, reviews, label):
        print("preprocess:", label)
        preprocessed_reviews = {}
        for review in reviews:
            cleaned_tokens = self.clean_text(review)
            self.processed_reviews[label].append(cleaned_tokens)

    def clean_and_save(self, output_file):
        # Process positive reviews
        print("folder in the class", self.folder)
        print("\npos")
        pos_reviews = self.load_reviews('pos')
        print("size of pos reviews", len(pos_reviews))
        self.preprocess_reviews(pos_reviews, 'pos')

        # Process negative reviews
        print("\nneg")
        neg_reviews = self.load_reviews('neg')
        print("size of neg reviews", len(neg_reviews))
        self.preprocess_reviews(neg_reviews, 'neg')

        # Combine positive and negative reviews
        print("combine")
        # for label, words in self.processed_reviews.items():
        #     print(label, ":")
        #     for word_list in words:
        #         print(word_list)

        print(len(self.processed_reviews['pos']), len(self.processed_reviews['neg']))

        # Save preprocessed reviews to a pickle file
        with open(output_file, 'w') as f:
            f.write(json.dumps(self.processed_reviews))
        print("Preprocessing and saving complete.")