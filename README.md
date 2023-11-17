# Reverse_Dictionary

This task is a ranking problem in which, given a definition, the task is to retrieve a ranked list of the most relevant words. 

## Dataset ##
First, We have used [3D-EX](https://github.com/F-Almeman/3D-EX/tree/main), , a unified resource containing several dictionaries in the format of (Term, Definition, Example, Source), to retrieve all the definitions along with their corresponding terms and sources, creating <definition, [list_of_terms_defined_by_that_definition], [list_of_sources]> tuples.  [definitions_dataset.csv](https://drive.google.com/uc?export=download&id=1Xhi_3OH1axN3Ch2hzTqFJX0ji1DBbd8I). 

```
python3 get_definitions.py -d 3dex.csv -o datasets
```
-d: input file (dataset) <br/>
-o: output folder <br/>

## Embeddings ##
Then, we computed definitions and terms embeddings using SBERT model. 

```
python3 get_embeddings.py -d definitions_dataset.csv -o datasets 
```
-d: input file (dataset) <br/>
-o: output folder <br/>
-m: SBERT model (default = "all-MiniLM-L6-v2" )


## Splitting ##
Two splits are created from the new dataset that includes definitions and terms embeddings which are Random_Split and Source_Split. Source_Split splits the datasets based on the specidied source ('MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition')

```
python3 split_dataset.py -d embed_dataset.csv -o datasets -s "WordNet"
```
-d: input file (dataset) <br/>
-o: output folder <br/>
-s: split type (default = "random" )

## Reverse Dictionary (RD) Experiment ##
Two splits are created from the new dataset that includes definitions and terms embeddings which are Random_Split and Source_Split. Source_Split splits the datasets based on the specidied source ('MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition')

```
python3 run_rd.py -d embed_dataset.csv -o datasets -s "WordNet"
```
-d: input file (dataset) <br/>
-o: output folder <br/>
-s: split type (default = "random" )

