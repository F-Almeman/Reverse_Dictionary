import torch
import pandas as pd
import ast
import argparse
from tqdm import tqdm
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objs as go
from torch.utils.data import DataLoader
import numpy as np

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
    
    args = parser.parse_args()

    model_name = 'args.model
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


    train_ds = Dataset.from_pandas(traindf)
    val_ds = Dataset.from_pandas(valdf)
    train_ds = train_ds.map(compute_labels, batched=True)
    val_ds = val_ds.map(compute_labels, batched=True)
    train_ds = train_ds.remove_columns(['TERMS'])
    val_ds = val_ds.remove_columns(['TERMS'])

    loss = losses.MSELoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"map_model_{model_name}_{args.source_name}",
        num_train_epochs=10,
        save_steps=1000000,  # Set a large number to avoid saving intermediate checkpoints
        save_strategy='epoch',  # Save checkpoints only at the end of each epoch
        save_total_limit=1,  # Keep only the most recent checkpoint
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss
    )

    # Train the model
    trainer.train()
