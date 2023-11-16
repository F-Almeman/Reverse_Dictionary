import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np
import random

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset_file',help='The main dataset (term, definition, example, source)',required=True)
  parser.add_argument('-o','--output_path',help='Path to output files (train/test/val)',required=True)

  args = parser.parse_args()
        
  # Read the dataset file
  dataset = pd.read_csv(args.dataset_file, engine='python', na_values = [''], keep_default_na=False)


  # Extract sources
  sources = ['MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition' ]
  dataset_list = []
  for idx in range(len(dataset)):
    l = []
    for s in sources:
      if s in dataset.iloc[idx].DATASET_NAME:
        l.append(s)
    dataset_list.append(set(l))

  dataset['DATASET_SET'] = dataset_list

  # Create a dataframe of (definition, [list of terms], [list of sources])
  new_dataset = {}

  for index, row in dataset.iterrows():
    definition = row['DEFINITION']
    word = row['TERM']
    source_list = list(row['DATASET_SET'])

    # Check if definition is already in the new_dataset dictionary
    if definition in new_dataset:
        new_dataset[definition]['TERMS_LIST'].append(word)
        new_dataset[definition]['SOURCE'].extend(source_list)
    else:
        new_dataset[definition] = {'TERMS_LIST': [word], 'SOURCE': source_list}

  # Create a list of tuples from the new_data dictionary
  new_data_list = [(key, value['TERMS_LIST'], value['SOURCE']) for key, value in new_dataset.items()]

  # Create a new dataframe from the list of tuples
  new_df = pd.DataFrame(new_data_list, columns=['DEFINITION', 'TERMS', 'SOURCES'])
  
  new_df.to_csv(os.path.join(args.output_path, "definitions.csv"), index = False, header=True)
