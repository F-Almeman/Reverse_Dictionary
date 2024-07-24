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

def extract_embeddings(model, dataset, num_samples=10):
    word_embeddings = []
    def_embeddings = []
    terms = []
    definitions = []
    for i, row in dataset[:num_samples].iterrows():
        word_embedding = model.encode(row['TERMS'])
        def_embedding = model.encode(row['DEFINITION'])
        word_embeddings.append(word_embedding)
        def_embeddings.append(def_embedding)
        terms.append(row['TERMS'])
        definitions.append(row['DEFINITION'])
    return np.array(word_embeddings), np.array(def_embeddings), terms, definitions

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def plot_embeddings(word_embeddings, def_embeddings, terms, definitions, title, filename):
    tsne = TSNE(n_components=2, random_state=42, perplexity=4)
    all_embeddings = np.concatenate([word_embeddings, def_embeddings])
    tsne_results = tsne.fit_transform(all_embeddings)
    
    word_points = tsne_results[:len(word_embeddings)]
    def_points = tsne_results[len(word_embeddings):]

    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly

    for i in range(len(word_points)):
        color = colors[i % len(colors)]
        similarity = cosine_similarity(word_embeddings[i], def_embeddings[i])
        fig.add_trace(go.Scatter(
            x=[word_points[i][0]],
            y=[word_points[i][1]],
            mode='markers',
            marker=dict(symbol='circle', size=10, color=color),
            name=f'Word {i + 1}',
            hoverinfo='text',
            hovertext=f'Word: {terms[i]}'
        ))
        fig.add_trace(go.Scatter(
            x=[def_points[i][0]],
            y=[def_points[i][1]],
            mode='markers',
            marker=dict(symbol='star', size=12, color=color),
            name=f'Definition {i + 1}',
            hoverinfo='text',
            hovertext=f'Definition: {definitions[i]}'
        ))
        fig.add_trace(go.Scatter(
            x=[word_points[i][0], def_points[i][0]],
            y=[word_points[i][1], def_points[i][1]],
            mode='lines',
            line=dict(color=color),
            showlegend=False
        ))
        fig.add_annotation(
            x=(word_points[i][0] + def_points[i][0]) / 2,
            y=(word_points[i][1] + def_points[i][1]) / 2,
            text=f'{similarity:.2f}',
            showarrow=False,
            font=dict(color=color)
        )

    fig.update_layout(title=title, showlegend=True)
    fig.write_html(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',  help='Model type', default='all-MiniLM-L6-v2')
    parser.add_argument('-s', '--source_name', help='source name', required=True)
    parser.add_argument('-t', '--train_file', help='train file', required=True)
    parser.add_argument('-v', '--valid_file', help='valid file', required=True)
    
    args = parser.parse_args()

    model_name = 'sentence-transformers/'+ args.model
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
