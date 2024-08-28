import torch
import pandas as pd
import ast
import argparse
from tqdm import tqdm
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, InputExample
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objs as go
from torch.utils.data import DataLoader
import numpy as np
import random
from datasets import Dataset

def create_pairs(df):
    sentence1 = []
    sentence2 = []
    labels = []

    for index, row in df.iterrows():
        positive_example = row['TERMS']
        sentence1.append(row['DEFINITION'])
        sentence2.append(positive_example)
        labels.append(1)  # Positive pair

        negative_example = random.choice(df[df['DEFINITION'] != row['DEFINITION']]['TERMS'].values)
        sentence1.append(row['DEFINITION'])
        sentence2.append(negative_example)
        labels.append(0)  # Negative pair

    return Dataset.from_dict({
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": labels,
    })
    

def create_triplets(df):
    all_terms = df.TERMS.tolist()
    triplets = []
    terms = df['TERMS'].tolist()
    for _, row in df.iterrows():
        anchor = row['DEFINITION']
        positive = row['TERMS']
        negative = random.choice(all_terms)
        if anchor.strip() and positive.strip() and negative.strip():
            triplets.append((anchor, positive, negative))
    return triplets
    
def compute_labels(batch):
    return {
        "label": model.encode(batch["TERMS"]),
    }
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',  help='Model type', default='all-MiniLM-L6-v2')
    parser.add_argument('-s', '--source_name', help='source name', required=True)
    parser.add_argument('-t', '--train_file', help='train file', required=True)
    parser.add_argument('-v', '--valid_file', help='valid file', required=True)
    parser.add_argument('-l', '--loss_function', help=' loss function', default="MSELoss")
    
    
    args = parser.parse_args()

    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name)

    traindf = pd.read_csv(args.train_file, na_values=[''], keep_default_na=False, index_col=False)
    valdf = pd.read_csv(args.valid_file, na_values=[''], keep_default_na=False, index_col=False)
    
    
    # train
    traindf['TERMS'] = traindf['TERMS'].apply(ast.literal_eval)
    traindf['EXAMPLES'] = traindf['EXAMPLES'].apply(ast.literal_eval)
    traindf['SOURCES'] = traindf['SOURCES'].apply(ast.literal_eval)
    traindf = traindf.explode('TERMS').reset_index(drop=True)

    if 'DEFINITION' not in traindf.columns or 'TERMS' not in traindf.columns:
        raise ValueError("The input CSV file must contain 'DEFINITION' and 'TERMS' columns")

    traindf = traindf[['DEFINITION', 'TERMS']]
    
    # valid
    valdf['TERMS'] = valdf['TERMS'].apply(ast.literal_eval)
    valdf['EXAMPLES'] = valdf['EXAMPLES'].apply(ast.literal_eval)
    valdf['SOURCES'] = valdf['SOURCES'].apply(ast.literal_eval)
    valdf = valdf.explode('TERMS').reset_index(drop=True)

    if 'DEFINITION' not in valdf.columns or 'TERMS' not in valdf.columns:
        raise ValueError("The input CSV file must contain 'DEFINITION' and 'TERMS' columns")

    valdf = valdf[['DEFINITION', 'TERMS']]

    loss_fun = args.loss_function
    
    if loss_fun == "ContrastiveLoss":
      loss = losses.ContrastiveLoss(model=model)
      
      # Create pairs for contrastive learning
      train_dataset = create_pairs(traindf)
      val_dataset = create_pairs(valdf)
     
    elif loss_fun == "TripletLoss":
      loss = losses.TripletLoss(model=model)
      # Create triplets
      train_triplets = create_triplets(traindf)
      val_triplets = create_triplets(valdf)

      train_dataset = Dataset.from_dict({
        'anchor': [triplet[0] for triplet in train_triplets],
        'positive': [triplet[1] for triplet in train_triplets],
        'negative': [triplet[2] for triplet in train_triplets]
      })
      val_dataset = Dataset.from_dict({
        'anchor': [triplet[0] for triplet in val_triplets],
        'positive': [triplet[1] for triplet in val_triplets],
        'negative': [triplet[2] for triplet in val_triplets]
      })    
    
    elif loss_fun == "MSEloss":
      loss = losses.MSELoss(model=model)
      train_dataset = Dataset.from_pandas(traindf)
      val_dataset = Dataset.from_pandas(valdf)
      train_dataset = train_dataset.map(compute_labels, batched=True)
      val_dataset = val_dataset.map(compute_labels, batched=True)
      train_dataset = train_dataset.remove_columns(['TERMS'])
      val_dataset = val_dataset.remove_columns(['TERMS'])
    
    print(f"train_size = {len(train_dataset)}")
    print(f"valid_size = {len(val_dataset)}")
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"{loss_fun}_{model_name}_{args.source_name}",
        num_train_epochs=50,
        save_steps=1000000,  # Set a large number to avoid saving intermediate checkpoints
        save_strategy='epoch',  # Save checkpoints only at the end of each epoch
        save_total_limit=1,  # Keep only the most recent checkpoint
    )

    trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    callbacks=[]  # Disable any default callbacks
    )

    # Train the model
    trainer.train()
