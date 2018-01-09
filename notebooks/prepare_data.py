import pandas as pd
import sqlite3
from os import path
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from lib.data_preprocessors import scale

db_path = path.join('..', 'data', 'iam-database', 'iam_words.db')
data_dir_path = path.join('..', 'data', 'iam-database')
# There are 115,320 words in total.
# Taking the first 92,256 (80%) for training
frame = pd.read_sql_query('SELECT * FROM words_index order by word_id limit 92256', sqlite3.connect(db_path))
# word_id:    TEXT,
# image_path: TEXT,
# status:     TEXT,
# gray_level: INTEGER,
# tokens:     TEXT
target_size = (200, 100)
images = []
texts = []
for i in tqdm(range(len(frame))):
    row = frame.iloc[i]
    image_path = path.join(data_dir_path, row.image_path)
    try:
        image = Image.open(image_path)
        image = image.point(lambda level: 255 if level > row.gray_level else 0)

        image = scale(image, target_size)
        images.append(np.array(image))
        texts.append(row.tokens)
    except OSError as e:
        print ('Error: ' + str(e))

images = np.stack(images, axis=0)
#print (images.shape)
#plt.imshow(images[5], cmap='binary')
#plt.show()
save_path = path.join(data_dir_path, 'train_words_images.npz')
np.savez(save_path, images)
print ('Saved image data as an array of size %s to "%s"' % (str(images.shape), path.abspath(save_path)))

save_path = path.join(data_dir_path, 'train_words_text.json')
json.dump(texts, open(save_path, 'w'))
print ('Saved transcribed text to %s' % path.abspath(save_path))