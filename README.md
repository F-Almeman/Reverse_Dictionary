# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

[compute_mrr.py](https://github.com/F-Almeman/Reverse_Dictionary/blob/main/compute_mrr.py) computes the Mean Reciprocal Rank (MRR), which rewards the position of the first correct result in a ranked list of outcomes, using different encoders (it supports Instructor encoder too). <br />
-m: model type (defult="all-MiniLM-L6-v2").<br /> 
-d: data input file (WORD, DEFINITION).<br /> 
-wi: word instruction (defult="no").<br /> 
-di: definition instruction (defult="no").<br /> 
```
python3 src/reverse_dictionary.py -m model -d random_test.csv -wi "no" -di "Represent this dictionary definition" 
```
