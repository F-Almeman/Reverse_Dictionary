import pandas as pd
import argparse
import os
import sys
import csv
from ast import literal_eval
from sentence_transformers import SentenceTransformer,util
import numpy as np
import ast
import torch
from pathlib import Path
import re

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
    
    
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset',help='The dataset',required=True)
  parser.add_argument('-e','--eval_method',help='The evaluation method',required=True)
  parser.add_argument('-o', '--output_csv', help='Output CSV file', required=True)
  

  
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
  if re.match(r'^p\d+$', args.eval_method):
    k = int(args.eval_method[1:])
    result =  np.mean(dataset['HITS'].apply(lambda x: precision_at_k(x, k)).tolist())
    
  output_data = {
        'dataset': [Path(args.dataset).stem],
        'method': [args.eval_method],
        'result': [result]
    }

  output_df = pd.DataFrame(output_data)

  # Check if the CSV file already exists
  csv_file_path = args.output_csv
  if Path(csv_file_path).is_file():
    # If the file exists, append the new result as a new row
    output_df.to_csv(csv_file_path, mode='a', header=False, index=False)
  else:
    # If the file does not exist, create a new file and write the header
    output_df.to_csv(csv_file_path, index=False)
