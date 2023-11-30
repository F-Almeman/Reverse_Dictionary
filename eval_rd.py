import pandas as pd
import argparse
import os
import sys
import csv
from ast import literal_eval
import numpy as np
import ast
import torch

# Evaluation metric
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset',help='The dataset',required=True)
  parser.add_argument('-e','--eval_method',help='The evaluation method',required=True)

  
  args = parser.parse_args()
  
  maxInt = sys.maxsize

  while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
      csv.field_size_limit(maxInt)
      break
    except OverflowError:
      maxInt = int(maxInt/10)
        
  # Read the dataset
  dataset = pd.read_csv(args.dataset, engine='python', na_values = [''], keep_default_na=False)
  
  # Convert the string representation of lists to actual lists
  dataset['HITS'] = dataset['HITS'].apply(lambda x: [int(i) for i in ast.literal_eval(x)])

  
  if args.eval_method == "mrr":
    result = mean_reciprocal_rank(dataset['HITS'].to_list())
    
  with open('output.txt', 'w') as file:
    file.write(f"Dataset: {args.dataset}\n")
    file.write(f"Evaluation method: {args.eval_method}\n")
    file.write(f"Result: {result}\n")
    
