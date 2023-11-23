import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np
from sentence_transformers import  SentenceTransformer
from pathlib import Path
import torch

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-i','--input_file',help='The input txt file',required=True)
  parser.add_argument('-o','--output_path',help='Path to the output file',required=True)
  parser.add_argument('-m', '--model', help='Model type', default='all-MiniLM-L6-v2')

  args = parser.parse_args()
        
  # Read the input file
  with open(args.input_file, 'r') as file:
    data = [line.strip() for line in file]

  # Load a pre-trained SBERT model
  model = SentenceTransformer('sentence-transformers/'+ args.model)

  # Compute embeddings 
  embeddings = model.encode(data, convert_to_tensor=True)

  # Save the embeddings to a numpy file
  np.save(os.path.join(args.output_path, f"{Path(args.input_file).stem}_{args.model}.npy"), embeddings.numpy())
