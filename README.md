# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

## Dataset ##
First, We have used [3D-EX](https://github.com/F-Almeman/3D-EX/tree/main), , a unified resource containing several dictionaries in the format of (Term, Definition, Example, Source), to retrieve all the definitions along with their corresponding terms and sources, creating <definition, [list_of_terms_defined_by_that_definition], [list_of_sources]> tuples.s [definitions.csv](https://drive.google.com/uc?export=download&id=1Xhi_3OH1axN3Ch2hzTqFJX0ji1DBbd8I). 


```
python3 get_definitions.py -d 3dex.csv -o datasets
```
