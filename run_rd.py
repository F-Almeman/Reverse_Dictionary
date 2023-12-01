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

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-s','--dataset_split',help='The dataset',required=True)
  parser.add_argument('-t','--terms_file',help='The terms file',required=True)
  parser.add_argument('-te','--terms_embeddings_file',help='The terms embeddings file',required=True)
  parser.add_argument('-d','--definitions_file',help='The definitions file',required=True)
  parser.add_argument('-de','--definitions_embeddings_file',help='The definitions embeddings file',required=True)
  parser.add_argument('-o','--output_path',help='Path to the output file',required=True)
  parser.add_argument('-k','--number_terms',help='The number of retrieved terms',required=True, type=int)


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
        
  # Read the files
  dataset = pd.read_csv(args.dataset_split, engine='python', na_values = [''], keep_default_na=False)
  dataset['TERMS'] = dataset['TERMS'].apply(ast.literal_eval)
  dataset['SOURCES'] = dataset['SOURCES'].apply(ast.literal_eval)

  with open(args.terms_file, 'r') as file:
    terms = [line.strip() for line in file]

  with open(args.definitions_file, 'r') as file:
    definitions = [line.strip() for line in file]
   
  definitions = [d.strip() for d in definitions]
  #terms = [t.strip() for t in terms]
  
  # Load the embeddings files
  terms_embeddings = np.load(args.terms_embeddings_file)
  def_embeddings = np.load(args.definitions_embeddings_file)

  top_k = args.number_terms
  hits_column = []
  pred_terms_column  = []
  
  #dataset = dataset.sample(100)

  for idx in range(len(dataset)):
    print(idx)
    definition = dataset.DEFINITION.iloc[idx].strip()
    gold_terms = dataset.TERMS.iloc[idx]
    index_of_def = definitions.index(definition)
    d_embedding = def_embeddings[index_of_def]

    t_embeddings = []
    for term in gold_terms:
      index_of_term = terms.index(term)
      t_embeddings.append(terms_embeddings[index_of_term])
      
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(d_embedding, terms_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    hits = []
    pred_terms = []
    
    for score, idx in zip(top_results[0], top_results[1]):
      predicted_term = terms[idx]
      pred_terms.append(predicted_term)
      if predicted_term in gold_terms:
        hits.append(1)
      else:
        hits.append(0)

    hits_column.append(hits)
    pred_terms_column.append(pred_terms)

  dataset["HITS"] =  hits_column
  dataset["PREDICTED_TERMS"] = pred_terms_column
  dataset.to_csv(os.path.join(args.output_path,f"{Path(args.terms_embeddings_file).stem}_{Path(args.definitions_embeddings_file).stem}_rd_dataset.csv"), index = False, header=True)
 
