import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np
from sentence_transformers import  SentenceTransformer, util
import torch

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset_file',help='The main dataset (term, definition, example, source)',required=True)
  parser.add_argument('-o','--output_path',help='Path to output files (train/test/val)',required=True)
  parser.add_argument('-m', '--model', help='Model type', default='all-MiniLM-L6-v2')

  args = parser.parse_args()
        
  # Read the dataset file
  df = pd.read_csv(args.dataset_file, engine='python', na_values = [''], keep_default_na=False)

  embedder = SentenceTransformer('sentence-transformers/'+ args.model)

  term_embed_column  = []
  def_embed_column  = []

  for idx in range(len(df)):
    term = df.TERM.iloc[idx]
    definition = df.DEFINITION.iloc[idx]

    term_embedding = embedder.encode(term, convert_to_tensor=True)
    definition_embedding = embedder.encode(definition, convert_to_tensor=True)

    term_embed_column.append(term_embedding)
    def_embed_column.append(definition_embedding)

  df["TERM_EMBED"] =  term_embed_column
  df["DEF_EMBED"] = def_embed_column

  # Convert NumPy arrays to lists
  df['TERM_EMBED'] = df['TERM_EMBED'].apply(lambda x: x.tolist())
  df['DEF_EMBED'] = df['DEF_EMBED'].apply(lambda x: x.tolist())
  df.rename(columns={"DATASET_NAME": "SOURCE"}, inplace=True)

  df.to_csv(os.path.join(args.output_path, "embed_dataset.csv"), index = False, header=True)
