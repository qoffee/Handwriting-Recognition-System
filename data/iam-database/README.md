# IAM Database

1. Download `lines.tgz` and `words.tgz` from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
2. Extract `lines.tgz` to a sub-directory `lines`.
3. Extract `words.tgz` to a sub-directory `words`.
4. Download `lines.txt` and `words.txt` from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and save in current directory.
5. Run the script to create an index of the dataset.

```
$ python3 create_dataset_index.py
```