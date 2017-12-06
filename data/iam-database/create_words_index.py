#!/usr/bin/env python3
from os import path
import sqlite3

connection = sqlite3.connect('iam_words.db')
cursor = connection.cursor()

cursor.execute('DROP TABLE IF EXISTS words_index')
cursor.execute('''CREATE TABLE words_index (
                    word_id TEXT,
                    image_path TEXT,
                    status TEXT,
                    gray_level INTEGER,
                    tokens TEXT)''')

with open('words.txt', 'r') as handle:
    for line in handle:
        if line.startswith('#'):
            # In the data-file, lines starting with
            # the '#' symbol are comments.
            # Ignoring the comments
            continue
        
        # Each row of data has nine parts separated by a single space ' '.
        # The parts are:
        # - word_id
        # - status: result of word segmentation ('ok', 'er')
        # - graylevel: the gray-level value to binarize the image
        # - n_components
        # - x, y, w, h: bounding box around this word
        # - tokens: transcription for this word.  
        parts = line.strip().split(' ')
        word_id, status, gray_level, n_components, x, y, w, h = parts[:8]
        tokens = ' '.join(parts[8:])

        # Each word_id is of the format a-b-c-d
        # such that the corresponding image is located at
        # words/a/a-b/a-b-c-d.png
        first, second, third, fourth = word_id.split('-')
        outer = first
        inner = first + '-' + second
        file_name = first + '-' + second + '-' + third + '-' + fourth + '.png'
        image_path = path.join('words', outer, inner, file_name)
        values = (word_id, image_path, status, 
            int(gray_level), tokens)
        
        cursor.execute('INSERT INTO words_index VALUES(?, ?, ?, ?, ?)', values)

result = next(cursor.execute('SELECT COUNT(*) from words_index'))
print ('Inserted {:,d} rows.'.format(result[0]))
connection.commit()
connection.close()