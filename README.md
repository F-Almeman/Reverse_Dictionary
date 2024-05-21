# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

## Embeddings ##
In this work we used [3D-EX](https://github.com/F-Almeman/3D-EX/tree/main), a unified resource containing several dictionaries in the format of (Term, Definition, Example, Source). First, we computed the embeddings of all terms, definitions, and examples and save these vestors into numpy files.

```
python3 get_embeddings.py -i datasets/definitions.txt -o datasets 
```
-i: input text file (terms file, definitions file, or examples file) <br/>
-o: output folder <br/>
-m: model (default = "all-MiniLM-L6-v2" form SBERT)

## Dataset ##
From 3D-EX, we retrieved all the definitions along with their corresponding terms, examples and sources, creating <definition, [list_of_terms_defined_by_that_definition], [list_of_examples],  [list_of_sources]> tuples. [dataset.csv](https://drive.google.com/uc?export=download&id=1qSFITj_gTBe9DOxOo4udrXmIjBvIthFG). 

```
python3 get_definitions.py -d 3dex.csv -o datasets
```
-d: input file (dataset) <br/>
-o: output folder 


## Splitting ##
Two splits are created: Random_Split and Source_Split. Source_Split splits the dataset based on the specified source ('MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition')

```
python3 split_dataset.py -d datasets/dataset.csv -o datasets -s "WordNet"
```
-d: input file (embedings dataset) <br/>
-o: output folder <br/>
-s: split type (default = "random" )

## Reverse Dictionary (RD) Experiment ##
This section is to retrive the best K terms for each definition based on the similarity between the definition embedding and all terms embeddings.
```
python3 run_rd.py -d rd_dataset_random_test.csv -t terms.txt -te terms_all-MiniLM-L6-v2.npy -d definitions.txt -de definitions_all-MiniLM-L6-v2.npy -o datasets -k 5 
```
-s: split dataset <br/>
-t: terms file <br/>
-te: terms embeddings file <br/>
-d: definitions file <br/>
-de definitions embeddings file <br/>
-o: output folder <br/>
-k: number of best terms

## RD Evaluation ##
Different measures could be used to evaluate the RD task. 
```
python3 eval_rd.py -d terms_all-MiniLM-L6-v2_definitions_all-MiniLM-L6-v2_rd_dataset.csv -e "mrr"
```

-d: the dataset generated from the previous step  <br/>
-e: evaluation method which are mean_reciprocal_rank and precision_at_k ("mrr", "p1", "p3", "p5") <br/>


