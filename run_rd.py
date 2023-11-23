import pandas as pd
import argparse
import os
import sys
import csv
from ast import literal_eval
import numpy as np
import ast

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset_file',help='The dataset',required=True)
  parser.add_argument('-t','--terms_file',help='The terms file',required=True)
  parser.add_argument('-te','--terms_embeddings_file',help='The terms embeddings file',required=True)
  parser.add_argument('-d','--definitions_file',help='The definitions file',required=True)
  parser.add_argument('-de','--definitions_embeddings_file',help='The definitions embeddings file',required=True)
  parser.add_argument('-o','--output_path',help='Path to the output file',required=True)
  parser.add_argument('-k','--number_terms',help='The number of retrieved terms',required=True)


  args = parser.parse_args()
        
  # Read the files
  dataset = pd.read_csv(args.dataset_file, engine='python', na_values = [''], keep_default_na=False)
  dataset['TERMS_LIST'] = dataset['TERMS_LIST'].apply(ast.literal_eval)
  dataset['SOURCES_LIST'] = dataset['SOURCES_LIST'].apply(ast.literal_eval)

  with open(args.terms_file, 'r') as file:
    terms = [line.strip() for line in file]

  with open(args.definitions_file, 'r') as file:
    definitions = [line.strip() for line in file]
    
  # Load the embeddings files
  terms_embeddings = np.load(args.terms_embeddings_file)
  def_embeddings = np.load(args.definitions_embeddings_file)

  k = args.number_terms
  hits_column = []
  pred_terms_column  = []

  for idx in range(len(dataset)):
    definition = dataset.DEFINITION.iloc[idx]
    gold_terms = dataset.TERM.iloc[idx]
    index_of_def = definitions.index(query)
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
      pred_terms.append(predicted_word)
      if predicted_word in gold_terms:
        hits.append(1)
      else:
        hits.append(0)

    hits_column.append(hits)
    pred_terms_column.append(pred_terms)

dataset["HITS"] =  hits_column
dataset["PREDICTED_TERMS"] = pred_terms_column
dataset.to_csv(os.path.join(args.output_path, "rd_dataset.csv"), index = False, header=True)
  
  

