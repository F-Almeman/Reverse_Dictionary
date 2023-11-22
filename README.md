# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

## Embeddings ##
In this work we used [3D-EX](https://github.com/F-Almeman/3D-EX/tree/main), a unified resource containing several dictionaries in the format of (Term, Definition, Example, Source). First, we computed the embeddings of all terms, definitions, and examples and save these vestors into numpy files.

```
python3 get_embeddings.py -d definitions.csv -o datasets 
```
-d: input file (terms file, definitions file, or examples file) <br/>
-o: output folder <br/>
-m: model (default = "all-MiniLM-L6-v2" form SBERT)

## Dataset ##
From 3D-EX, we retrieved all the definitions along with their corresponding terms, examples and sources, creating <definition, [list_of_terms_defined_by_that_definition],[list_of_examples],  [list_of_sources]> tuples. [dataset.csv](https://drive.google.com/uc?export=download&id=11B25YeDUkIhPIqXCrxvHIoU-ovSw0W4s). 

```
python3 get_definitions.py -d 3dex.csv -o datasets
```
-d: input file (dataset) <br/>
-o: output folder <br/


## Splitting ##
Two splits are created from the new dataset that includes definitions and terms embeddings which are Random_Split and Source_Split. Source_Split splits the dataset based on the specified source ('MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition')

```
python3 split_dataset.py -d embed_dataset.csv -o datasets -s "WordNet"
```
-d: input file (embedings dataset) <br/>
-o: output folder <br/>
-s: split type (default = "random" )

## Reverse Dictionary (RD) Experiment ##
This section is to retrive the best K terms for each definition based on the similarity between the definition embedding and all terms embeddings.
```
python3 run_rd.py -d dataset.csv -k 5 ....
```
-d: input file (split dataset) <br/>
-o: output folder <br/>
-k: number of best terms

