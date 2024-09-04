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

def corrupt_definition(definition, mask_token='[MASK]', corruption_percentage=0.15, exclude_indices=set()):
    # Tokenize the definition at the word level
    words = word_tokenize(definition)
    num_words = len(words)
    
    # Calculate the number of words to mask
    num_to_mask = max(1, int(corruption_percentage * num_words))
    
    # If there are fewer words than the number we want to mask, adjust num_to_mask
    num_to_mask = min(num_to_mask, num_words - len(exclude_indices))
    
    # Create a list of available indices excluding the ones already masked
    available_indices = list(set(range(num_words)) - exclude_indices)
    
    # Randomly choose words to mask
    mask_indices = random.sample(available_indices, num_to_mask)
    
    # Create a corrupted version of the definition
    corrupted_words = [mask_token if i in mask_indices else word for i, word in enumerate(words)]
    corrupted_definition = ' '.join(corrupted_words)
    
    return corrupted_definition, set(mask_indices)

def corrupt_and_augment_data(df, mask_token='[MASK]', corruption_percentage=0.15):
    augmented_data = []
    
    for index, row in df.iterrows():
        original_definition = row['DEFINITION']
        terms = row['TERMS']
        
        # Generate the first corrupted version
        corrupted_definition_v1, mask_indices_v1 = corrupt_definition(
            original_definition, mask_token, corruption_percentage
        )
        
        # Generate the second corrupted version, ensuring different words are masked
        corrupted_definition_v2, _ = corrupt_definition(
            original_definition, mask_token, corruption_percentage, exclude_indices=mask_indices_v1
        )
        
        # Append original and corrupted versions to the augmented data list
        augmented_data.append({'DEFINITION': original_definition, 'TERMS': terms})
        augmented_data.append({'DEFINITION': corrupted_definition_v1, 'TERMS': terms})
        augmented_data.append({'DEFINITION': corrupted_definition_v2, 'TERMS': terms})
    
    # Convert the augmented data list to a DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    
    return augmented_df

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

    if 'DEFINITION' not in traindf.columns or 'TERMS' not in traindf.columns:
        raise ValueError("The input CSV file must contain 'DEFINITION' and 'TERMS' columns")
        
    if 'DEFINITION' not in valdf.columns or 'TERMS' not in valdf.columns:
        raise ValueError("The input CSV file must contain 'DEFINITION' and 'TERMS' columns")
    
    
     # train
    traindf['TERMS'] = traindf['TERMS'].apply(ast.literal_eval)
    augmented_traindf = corrupt_and_augment_data(traindf)
    augmented_traindf = augmented_traindf.explode('TERMS').reset_index(drop=True)

    # valid
    valdf['TERMS'] = valdf['TERMS'].apply(ast.literal_eval)
    augmented_valdf = corrupt_and_augment_data(valdf)
    augmented_valdf = augmented_valdf.explode('TERMS').reset_index(drop=True)

    loss_fun = args.loss_function
    
    if loss_fun == "ContrastiveLoss":
      loss = losses.ContrastiveLoss(model=model)
      
      # Create pairs for contrastive learning
      train_dataset = create_pairs(augmented_traindf)
      val_dataset = create_pairs(augmented_valdf)
     
    elif loss_fun == "TripletLoss":
      loss = losses.TripletLoss(model=model)
      # Create triplets
      train_triplets = create_triplets(augmented_traindf)
      val_triplets = create_triplets(augmented_valdf)

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
      train_dataset = Dataset.from_pandas(augmented_traindf)
      val_dataset = Dataset.from_pandas(augmented_valdf)
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
        evaluation_strategy='epoch',  # Evaluate at the end of each epoch
        logging_steps=10,  # Log every 10 steps
        logging_first_step=True,  # Log the first step
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
