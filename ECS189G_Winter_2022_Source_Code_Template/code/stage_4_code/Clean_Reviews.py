import os
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class Clean_Reviews:
    def __init__(self, folder):
        self.folder = folder

    def clean_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    def load_reviews(self, label):
        reviews = []
        subfolder = 'pos' if label == 'pos' else 'neg'
        folder_path = os.path.join(self.folder, subfolder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                review = file.read()
                reviews.append(review)
        return reviews

    def preprocess_reviews(self, reviews, label):
        preprocessed_reviews = []
        for review in reviews:
            cleaned_tokens = self.clean_text(review)
            preprocessed_reviews.append((cleaned_tokens, label))
        return preprocessed_reviews

    def clean_and_save(self, output_file):
        # Process positive reviews
        pos_reviews = self.load_reviews('pos')
        preprocessed_pos_reviews = self.preprocess_reviews(pos_reviews, 'pos')

        # Process negative reviews
        neg_reviews = self.load_reviews('neg')
        preprocessed_neg_reviews = self.preprocess_reviews(neg_reviews, 'neg')

        # Combine positive and negative reviews
        all_reviews = preprocessed_pos_reviews + preprocessed_neg_reviews

        # Save preprocessed reviews to a pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(all_reviews, f)

        print("Preprocessing and saving complete.")


# Example usage:
cleaner = Clean_Reviews('your_folder_name_here')
cleaner.clean_and_save('preprocessed_reviews_tokens.pkl')

