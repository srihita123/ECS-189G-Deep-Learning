from code.stage_4_code.Data_Clean_Classification import Clean_Reviews
import string

if 1:
    train_file_path = 'C:/WSL/ECS_189G/189G_Project/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/train'
    test_file_path = 'C:/WSL/ECS_189G/189G_Project/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/test'

    clean_train = Clean_Reviews(train_file_path)
    clean_train.clean_and_save('preprocessed_reviews_tokens_train.pkl')

    clean_test = Clean_Reviews(train_file_path)
    clean_test.clean_and_save('preprocessed_reviews_tokens_test.pkl')
