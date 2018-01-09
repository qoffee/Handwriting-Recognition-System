from lib.data_generators import WordsDatasetIterator
from lib.models import WordModel
from os import path
import numpy as np
import json
from datetime import datetime
    


images_path = path.join('..', 'data', 'iam-database', 'train_words_images.npz')
words_path = path.join('..', 'data', 'iam-database', 'train_words_text.json')
images = np.load(images_path)['arr_0']
words = json.load(open(words_path))
batch_size = 50
pool_size = 2
words_data = WordsDatasetIterator(images, words, batch_size=batch_size)

        
model = WordModel(pool_size=pool_size, vocabulary_size=words_data.get_vocabulary_size()+1)
model.fit_generator(words_data, steps_per_epoch=92250 // batch_size)
save_path = path.join('..', 'models', 'model', 'word_model_%s.h5' % str(datetime.now()))
model.save(save_path)