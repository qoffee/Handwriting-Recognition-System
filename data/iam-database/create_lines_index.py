#!/usr/bin/env python3
from os import path
import sqlite3

connection = sqlite3.connect('iam_lines.db')
cursor = connection.cursor()

cursor.execute('DROP TABLE IF EXISTS lines_index')
cursor.execute('''CREATE TABLE lines_index (
                    line_id TEXT,
                    image_path TEXT,
                    status TEXT,
                    gray_level INTEGER,
                    tokens TEXT)''')

with open('lines.txt', 'r') as handle:
    for line in handle:
        if line.startswith('#'):
            # In the data-file, lines starting with
            # the '#' symbol are comments.
            # Ignoring the comments
            continue
        
        # Each row of data has nine parts separated by a single space ' '.
        # The parts are:
        # - line_id
        # - status: result of word segmentation ('ok', 'err', 'notice')
        # - graylevel: the gray-level value to binarize the image
        # - n_components
        # - x, y, w, h: bounding box around this line
        # - tokens: transcription for this line. Word tokens are 
        #   separated by the '|' character.
        parts = line.strip().split(' ')
        line_id, status, gray_level, n_components, x, y, w, h = parts[:8]
        tokens = ' '.join(parts[8:])

        # Each line_id is of the format a-b-c
        # such that the corresponding image is located at
        # lines/a/a-b/a-b-c.png
        first, second, third = line_id.split('-')
        outer = first
        inner = first + '-' + second
        file_name = first + '-' + second + '-' + third + '.png'
        image_path = path.join('lines', outer, inner, file_name)
        values = (line_id, image_path, status, 
            int(gray_level), tokens.replace('|', ' '))
        
        cursor.execute('INSERT INTO lines_index VALUES(?, ?, ?, ?, ?)', values)

result = next(cursor.execute('SELECT COUNT(*) from lines_index'))
print ('Inserted {:,d} rows.'.format(result[0]))
connection.commit()
connection.close()