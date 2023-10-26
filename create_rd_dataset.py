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

  parser.add_argument('-d','--dataset_file',help='The main dataset (word, definition, example, source)',required=True)
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
  new_df = pd.DataFrame(new_data_list, columns=['DEFINITION', 'TERMS_LIST', 'SOURCE'])
  
  new_df['NUMBER_TERMS'] = new_df['TERMS_LIST'].str.len()
  new_df['NUMBER_SOURCES'] = new_df['SOURCE'].apply(lambda x: len(set(x)))
  
  # Random splits
  random_train, random_valid, random_test = \
              np.split(unified_dataset.sample(frac=1, random_state=42), 
                       [int(.6*len(unified_dataset)), int(.8*len(unified_dataset))])

  
  new_df.to_csv(os.path.join(args.output_path, "RD_dataset.csv"), index = False, header=True)
  random_train.to_csv(os.path.join(args.output_path, "random_train.csv"), index = False, header=True)
  random_valid.to_csv(os.path.join(args.output_path, "random_valid.csv"), index = False, header=True)
  random_test.to_csv(os.path.join(args.output_path, "random_test.csv"), index = False, header=True)
