from dataloader import *

subject = 'M15'

data_train, data_test, glove_train, glove_test, data_fine, data_fine_test, glove_fine, glove_fine_test = dataloader_sentence_word_split_new_matching_all_subjects(subject)

print(data_train.shape, data_test.shape)