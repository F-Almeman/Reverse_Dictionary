# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

# Dataset #
From [3D-EX](https://github.com/F-Almeman/3D-EX/tree/main), a unified resource containing several dictionaries in the format of (Term, Definition, Example, Source), we retrieved all the definitions along with their corresponding terms, examples and sources, creating <definition, [list_of_terms_defined_by_that_definition], [list_of_examples],  [list_of_sources]> tuples. [dataset.csv](https://drive.google.com/uc?export=download&id=1TdVx9Pk3SQ16vWkr8WBi6SLpKMV9tIm6)

```
python3 get_definitions.py -d 3dex.csv -o datasets
```
-d: input file (dataset) <br/>
-o: output folder 

## Splitting ##
Two splits are created: Random_Split (train, validation, and test) and Source_Split. Source_Split splits the dataset based on the specified source ('MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition')

```
python3 split_dataset.py -d datasets/dataset.csv -o datasets -s "WordNet"
```
-d: input file  <br/>
-o: output folder <br/>
-s: split type (default = "random" )

# Reverse Dictionary (RD) Experiment #

## Unsupervised Learning ##

## Embeddings ##
First, we compute the embeddings of all terms, definitions, and examples in 3D-EX and save these vestors into numpy files.

```
python3 get_embeddings.py -i datasets/definitions.txt -o datasets 
```
-i: input text file (terms file, definitions file, or examples file) <br/>
-o: output folder <br/>
-m: model (default = "all-MiniLM-L6-v2" form SBERT)


## RD ##

This script is to retrive the best K terms for each definition based on the similarity between the definition embedding and all terms embeddings.
```
python3 run_rd.py -d datasets/random_test.csv -t terms.txt -te terms_all-MiniLM-L6-v2.npy -d definitions.txt -de definitions_all-MiniLM-L6-v2.npy -o datasets -k 5 
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
python3 eval_rd.py -d "datasets/random_test_terms_all-MiniLM-L6-v2_definitions_all-MiniLM-L6-v2_rd_dataset.csv" -e "mrr" -o datasets/outputs/rd_resuls.csv

```

-d: the dataset generated from the previous step  <br/>
-e: evaluation method which are mean_reciprocal_rank and precision_at_k ("mrr", "p1", "p3", "p5") <br/>
-o: csv file to save the results

## Supervised Learning ##

## Model Fine-tuning ##
This script fine-tunes a model on a dataset of terms and definitions to improve their embeddings. It does this by training the model using a Mean Squared Error loss to minimize the distance between the embeddings of terms and their corresponding definitions.

Different measures could be used to evaluate the RD task. 
```
python3 train_model.py -m "thenlper/gte-large" -s "WordNet" -t "datasets/WordNet_train.csv" -v "datasets/WordNet_valid.csv"
```
-m: model (default = "all-MiniLM-L6-v2" form SBERT)
-s: Source dataset <br/>
-t: train split <br/>
-v: valid split <br/>

The saved model could be used to compute the embeddings of terms and definitions and these embedding are used then in the RD experiment (as in the above scripts)
