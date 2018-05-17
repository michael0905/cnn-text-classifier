import numpy as np
import re
import itertools
from collections import Counter
from tflearn.data_utils import VocabularyProcessor
from sklearn.model_selection import train_test_split
from utils.utils import initialize_vocab

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        # Load data from files
        positive_data_file = self.config.positive_data_file
        negative_data_file = self.config.negative_data_file
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]
        self.max_text_length = max([len(x.split(" ")) for x in x_text])
        # transform words to id
        vocab_processor = VocabularyProcessor(self.max_text_length)
        X = np.array(list(vocab_processor.fit_transform(x_text)))
        self.vocab_size = len(vocab_processor.vocabulary_)
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        self.train_input, self.valid_input, self.train_label, self.valid_label = train_test_split(X, y,
                                                                                    random_state=2018, test_size=0.1)
        print('training size: {}'.format(len(self.train_input)))

    def next_batch(self, batch_size, is_training=True):
        if is_training:
            idx = np.random.choice(len(self.train_input), batch_size)
            yield self.train_input[idx], self.train_label[idx]
        else:
            idx = np.random.choice(len(self.valid_input), batch_size)
            yield self.valid_input[idx], self.valid_label[idx]

    def sequence_length(self):
        return self.max_text_length

    def get_vocab_size(self):
        return self.vocab_size
