import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np
import random
import ast
from ast import literal_eval

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset_file',help='The dataset file',required=True)
  parser.add_argument('-o','--output_path',help='Path to output files',required=True)
  parser.add_argument('-s','--split_type',help='Split type (random or source split)',default="random")

  args = parser.parse_args()
        
  # Read the dataset file
  dataset = pd.read_csv(args.dataset_file, engine='python', na_values = [''], keep_default_na=False)
  
  # Random Split
  if args.split_type == "random":
    train, valid, test = \
              np.split(dataset.sample(frac=1, random_state=42),
                       [int(.6*len(dataset)), int(.8*len(dataset))])

    train.to_csv(os.path.join(args.output_path, "dataset_random_train.csv", header = True, index = False)
    valid.to_csv(os.path.join(args.output_path, "dataset_random_valid.csv", header = True, index = False)
    test.to_csv(os.path.join(args.output_path, "dataset_random_test.csv", header = True, index = False)
 
  # Source Split
  else:
    split = dataset[dataset['SOURCES'].str.contains(args.split_type)]
    split.to_csv(os.path.join(args.output_path, f"dataset_{args.split_type}.csv", header = True, index = False)
