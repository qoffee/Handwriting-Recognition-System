import pandas as pd
import sqlite3
from os import path

db_path = path.join('..', 'data', 'iam-database', 'iam_words.db')

# There are 115,320 words in total.
# Taking the first 92,256 (80%) for training
frame = pd.read_sql_query('SELECT * FROM words_index order by word_id limit 92256', sqlite3.connect(db_path))