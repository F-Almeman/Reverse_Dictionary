# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

## Dataset ##
We have used [3D-EX](https://drive.google.com/uc?export=download&id=1ZjuRUn6KZPaXMVYecZ5IYDIRiB5VuEsR), a unified resource containing several dictionaries,to create a list of <definition, [list_of_words_defined_by_that_definition], [list_of_sources]> tuples [RD_dataset.csv](https://drive.google.com/uc?export=download&id=1diYrlHgwt8Fi2BmryO4V4tZNBGSpx9jV). 


```
python3 create_rd_dataset.py -d 3dex.csv -s "random" -o datasets
```
