import numpy as np
import json
import random

class WordsDatasetIterator:
    def __init__(self, images, words, batch_size=32, max_word_length=21, downsample_factor=4):
        
        self.batch_size = batch_size
        self.max_word_length = max_word_length
        self.downsample_factor = downsample_factor
        self.n_items, self.image_height, self.image_width = images.shape
        
        # Reorder such that width is the temporal dimension
        self.input_data = np.transpose(images, (0, 2, 1))

        # Create Character-to-Index and
        # Index-to-Character tables
        joined = ''.join(words)
        characters_set = set(joined)
        characters = list(characters_set)
        characters = list(sorted(characters))
        
        # Add character to represent padding
        characters = ['_'] + characters
        self.index_to_character = { i: c for i, c in enumerate(characters)}
        self.character_to_index = { c: i for i, c in self.index_to_character.items() }

        # Translate words to indices
        self.ground_truth_labels = [self.word_to_indices(word) for word in words]

        # Start iterator index at zero
        self.indices = list(range(len(self.ground_truth_labels)))
        self.current_index = 0
        self.reset_index()
    
    def get_vocabulary_size(self):
        return len(self.index_to_character)

    def indices_to_word(self, indices):
        return [self.index_to_character[i] for i in indices]
    
    def word_to_indices(self, word):
        return [self.character_to_index[c] for c in word]
    
    def reset_index(self):
        self.current_index = 0
        random.shuffle(self.indices)

    def get_next_item(self):
        self.current_index += 1
        
        if self.current_index >= self.n_items:
            self.reset_index()

        i = self.indices[self.current_index]
        return self.input_data[i], self.ground_truth_labels[i]
    
    def __iter__(self):
        return self

    def __next__(self):
        X = np.zeros((self.batch_size, self.image_width, self.image_height, 1))
        y = np.zeros((self.batch_size, self.max_word_length))

        # if pool_size is 2
        # downsample_factor is 2**2 = 4
        input_lengths = np.ones((self.batch_size, 1)) * (self.image_width // self.downsample_factor - 2)
        label_lengths = np.zeros((self.batch_size, 1))
        
        for i in range(self.batch_size):
            while True:
                input_data, label = self.get_next_item()
                if len(label) > 0:
                    break
            X[i, :, :] = np.expand_dims(input_data, -1)
            y[i, 0:len(label)] = label
            label_lengths[i, 0] = len(label)

        inputs = {
            'the_input': X,
            'the_labels': y,
            'input_length': input_lengths,
            'label_length': label_lengths
        }

        outputs = { 'ctc': np.zeros([self.batch_size]) }
        return (inputs, outputs)


if __name__ == '__main__':
    from os import path
    
    images_path = path.join('..', 'data', 'iam-database', 'train_words_images.npz')
    words_path = path.join('..', 'data', 'iam-database', 'train_words_text.json')
    images = np.load(images_path)['arr_0']
    words = json.load(open(words_path))
    words_data = WordsDatasetIterator(images, words, batch_size=4)
    x, y = next(words_data)
    print ('x:\n')
    for k, v in x.items():
        print ('"%s": array of shape %s' % (k, str(v.shape)))
    print ('\ny:\n')
    for k, v in y.items():
        print ('"%s": array of shape %s' % (k, str(v.shape)))