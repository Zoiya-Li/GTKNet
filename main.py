"""
Main entry point for the GTKNet model training and evaluation.
This module handles the complete pipeline from data loading to model training and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pandas as pd
from model import GTModel, NodeMaskingTask, ComprehensiveLoss
from data_processing import (
    load_data, 
    split_data, 
    prepare_embedding, 
    build_embeddings_tensor,
    split_graph_into_subgraphs,
    TripletDataset
)
from training import (
    pretrain_self_supervised,
    save_model,
    run_model,
    evaluate,
    assign_pseudo_labels
)
import logging
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import subgraph

# Set environment variables for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_all_node_embeddings(model, data, device, batch_size=512):
    """
    Get embeddings for all nodes using batch processing to avoid memory overflow.

    Args:
        model (nn.Module): Trained model
        data (Data): PyG data object
        device (torch.device): Computation device
        batch_size (int): Number of nodes per evaluation batch

    Returns:
        np.ndarray: Node embeddings array of shape (num_nodes, embedding_dim)
    """
    model.eval()
    num_nodes = data.num_nodes
    embeddings = []
    
    # Ensure edge_index is on CPU
    edge_index_cpu = data.edge_index.cpu()

    with torch.no_grad():
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            # Create batch indices on CPU
            batch_indices = torch.arange(start, end)
            batch_nodes_tensor = batch_indices

            # Extract subgraph on CPU
            subset_edge_index, _ = subgraph(
                batch_nodes_tensor,
                edge_index_cpu,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )

            # Move required tensors to the specified device
            subset_edge_index = subset_edge_index.to(device)
            subset_x = data.x[batch_indices].to(device)

            # Run the model on the subgraph
            subset_embeddings, logits, _ = run_model(model, subset_x, subset_edge_index, logits_index=1)
            
            # Move embeddings to CPU and convert to NumPy
            subset_embeddings = subset_embeddings.cpu().numpy()
            embeddings.append(subset_embeddings)

            # Release memory
            del subset_x
            del subset_edge_index
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    return all_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to run the GTKNet pipeline.
    Handles data loading, model training, and evaluation.
    """
    # Set environment variables to optimize memory allocation (optional)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data paths
    num_subgraphs = 20
    GT_hidden_emb = 32
    feature_path = '../dataset/elliptic_bitcoin_dataset/txs_features.csv'
    label_path = '../dataset/elliptic_bitcoin_dataset/txs_classes.csv'
    edgelist_path = '../dataset/elliptic_bitcoin_dataset/txs_edgelist.csv'
    model_dir = 'elliptic_bitcoin_dataset'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.pt')
    results_path = os.path.join(model_dir, 'results.pt')
    embeddings_file = os.path.join(model_dir, 'KG_node_embeddings.csv')

    # Load data
    data, node_to_idx = load_data(feature_path, label_path, edgelist_path)
    
    # Create reverse mapping
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}
    
    # Record initial device information
    logging.info(f"Initial edge_index device: {data.edge_index.device}")
    logging.info(f"Initial features device: {data.x.device}")
    
    # Ensure data is on CPU
    data.x = data.x.cpu()
    data.edge_index = data.edge_index.cpu()
    data.y = data.y.cpu()
    
    # Split the dataset
    data = split_data(data)
    
    # Record data information
    logging.info(f"Number of nodes: {data.num_nodes}")
    logging.info(f"Number of edges: {data.edge_index.size(1)}")
    logging.info(f"Feature dimensions: {data.x.size(1)}")

    # Ensure data.y is torch.long type
    data.y = data.y.to(torch.long)

    # Print unique labels
    unique_labels = torch.unique(data.y)
    num_classes = len(unique_labels)

    # Move data.x, data.edge_index, and data.y to the device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    KG_embeddings_df = prepare_embedding({'edge_index': data.edge_index}, data, node_to_idx, device, embeddings_file, model_dir)

    # Split the graph into subgraphs
    subgraphs = split_graph_into_subgraphs(data, num_subgraphs=num_subgraphs)

    # Initialize the model
    model = GTModel(
        input_dim=data.x.size(1),
        hidden_dim=GT_hidden_emb,
        num_layers=3,
        dropout=0.3,
        graph_conv_type='sage',
        num_heads=4,
        num_classes=num_classes
    ).to(device)

    masking_task = NodeMaskingTask(embedding_dim=GT_hidden_emb).to(device)

    # Self-supervised pre-training
    logging.info("Starting self-supervised pre-training...")
    pretrain_self_supervised(model, masking_task, subgraphs, epochs=50, device=device)
    logging.info("Self-supervised pre-training completed.")

    # Initialize the optimizer and learning rate scheduler
    loss_function = ComprehensiveLoss(triplet_margin=20.0).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(masking_task.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Initialize TripletDataset and DataLoader
    triplet_dataset = TripletDataset(data.y.cpu().numpy())

    # Create batches using DataLoader
    triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)

    # Self-training parameters
    best_val_acc = 0
    patience_counter = 10
    patience = 10
    num_epochs = 100
    self_training_iterations = 50
    pseudo_threshold = 0.80

    for iteration in range(self_training_iterations):
        logging.info(f"Self-training iteration {iteration + 1}/{self_training_iterations}")
        for epoch in range(num_epochs):
            model.train()
            masking_task.train()
            total_loss = 0.0
            total_cls_loss = 0.0
            total_triplet_loss = 0.0
    
            for batch_idx, batch in enumerate(triplet_loader, 1):
                anchor_idx, positive_idx, negative_idx = batch
                anchor_idx = anchor_idx.tolist()
                positive_idx = positive_idx.tolist()
                negative_idx = negative_idx.tolist()
    
                # Collect all relevant node indices
                batch_nodes = list(set(anchor_idx + positive_idx + negative_idx))
                
                # Move all relevant tensors to CPU for processing
                edge_index_cpu = data.edge_index.cpu()
                batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long)
                
                # Extract subgraph on CPU
                subset_edge_index, _ = subgraph(
                    batch_nodes_tensor,
                    edge_index_cpu,
                    relabel_nodes=True,
                    num_nodes=data.num_nodes
                )

                # Move results back to the device
                subset_edge_index = subset_edge_index.to(device)
                subset_x = data.x[batch_nodes].to(device)
                
                # Add debugging logs
                logging.debug(f"Batch nodes device: {batch_nodes_tensor.device}")
                logging.debug(f"Edge index device: {subset_edge_index.device}")
                logging.debug(f"Features device: {subset_x.device}")
    
                # Run the model on the subgraph
                embeddings, logits, attn_weights = run_model(model, subset_x, subset_edge_index, logits_index=1)
    
                # Get relative indices
                anchor_relative = [batch_nodes.index(idx) for idx in anchor_idx]
                positive_relative = [batch_nodes.index(idx) for idx in positive_idx]
                negative_relative = [batch_nodes.index(idx) for idx in negative_idx]
    
                # Extract corresponding logits
                logits_anchor = logits[anchor_relative]
                logits_positive = logits[positive_relative]
                logits_negative = logits[negative_relative]
    
                # Get original labels
                labels_anchor = data.y[anchor_idx].to(device)
                labels_positive = data.y[positive_idx].to(device)
                labels_negative = data.y[negative_idx].to(device)
    
                # Calculate triplet embeddings
                embeddings_anchor = embeddings[anchor_relative]
                embeddings_positive = embeddings[positive_relative]
                embeddings_negative = embeddings[negative_relative]
    
                # Calculate loss
                loss, loss_cls, loss_triplet = loss_function(
                    logits_anchor, labels_anchor,
                    logits_positive, labels_positive,
                    logits_negative, labels_negative,
                    embeddings_anchor, embeddings_positive, embeddings_negative
                )
    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(masking_task.parameters(), max_norm=1.0)
    
                total_loss += loss.item()
                total_cls_loss += loss_cls.item()
                total_triplet_loss += loss_triplet.item()
    
            avg_loss = total_loss / len(triplet_loader)
            avg_cls_loss = total_cls_loss / len(triplet_loader)
            avg_triplet_loss = total_triplet_loss / len(triplet_loader)
    
            logging.info(f"[Self-training iteration {iteration + 1}] Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, "
                         f"Cls Loss: {avg_cls_loss:.4f}, Triplet Loss: {avg_triplet_loss:.4f}")
    
            # Validate
            val_acc = evaluate(model, data, device)
            scheduler.step(val_acc)
    
            logging.info(f"Validation accuracy: {val_acc:.4f}")
    
            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_model(model, model_path)
                logging.info(f"[Self-training iteration {iteration + 1}] Epoch {epoch + 1}: New best validation accuracy {val_acc:.4f}, model saved.")
            else:
                patience_counter += 1
                logging.info(f"[Self-training iteration {iteration + 1}] Epoch {epoch + 1}: Validation accuracy did not improve ({val_acc:.4f})")
                if patience_counter >= patience:
                    logging.info(f"Early stopping at iteration {iteration + 1}, epoch {epoch + 1}")
                    break
    
            # Generate pseudo-labels and get the updated unknown_mask
            pseudo_labels, updated_unknown_mask = assign_pseudo_labels(model, data, threshold=pseudo_threshold, device=device)
            logging.info(f"Generated pseudo-labels: {len(pseudo_labels)}")
    
            if pseudo_labels:
                # Collect newly assigned labels
                new_labels = [(idx, label) for idx, label in pseudo_labels.items()]
    
                # Update labels
                for idx, label in new_labels:
                    if label < 0 or label >= num_classes:
                        logging.error(f"Node {idx} pseudo-label {label} is out of range!")
                        continue
                    data.y[idx] = label  # Ensure label is an integer
    
                # Print the updated label distribution
                label_counts = torch.bincount(data.y)
                logging.info(f"Updated label distribution: {label_counts}")
    
                # Add new triplets to the dataset
                triplet_dataset.add_triplets_from_new_labels(new_labels)
    
                # Save node txId, embeddings, and labels
                logging.info("Extracting all node embeddings...")
                all_embeddings = get_all_node_embeddings(model, data, device, batch_size=512)
                logging.info("Node embeddings extraction completed.")
    
                # Get labels
                labels = data.y.cpu().numpy()
    
                # Create a data dictionary
                data_dict = {'txId': [idx_to_node[idx] for idx in range(data.num_nodes)]}
    
                # Add embeddings
                embedding_dim = all_embeddings.shape[1]
                for emb_dim in range(embedding_dim):
                    data_dict[f'embedding_{emb_dim}'] = all_embeddings[:, emb_dim]
    
                # Add labels
                data_dict['label'] = labels
    
                # Create a DataFrame
                df = pd.DataFrame(data_dict)
    
                # Optionally: check the first few rows of the DataFrame
                logging.info(f"DataFrame shape: {df.shape}")
                logging.info(f"DataFrame first 5 rows:\n{df.head()}")
    
                # Save to CSV
                embeddings_with_labels_path = os.path.join(model_dir, 'node_embeddings_with_labels.csv')
                df.to_csv(embeddings_with_labels_path, index=False)
                logging.info(f"Node embeddings and labels saved to {embeddings_with_labels_path}")
            else:
                logging.info("No pseudo-labels generated, stopping self-training.")
                break

if __name__ == "__main__":
    main()