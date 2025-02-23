"""
Training utilities for the GTKNet model.
This module contains functions for model training, evaluation, and pseudo-labeling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import logging
from torch_geometric.utils import subgraph

def pretrain_self_supervised(model, masking_task, subgraphs, epochs=50, 
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Pre-train the model using self-supervised learning with node masking.

    Args:
        model (nn.Module): The main model
        masking_task (nn.Module): Node masking task module
        subgraphs (list): List of subgraphs for training
        epochs (int): Number of training epochs
        device (torch.device): Computation device
    """
    model.train()
    masking_task.train()
    optimizer = torch.optim.Adam(masking_task.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler()  # Initialize gradient scaler
    
    for epoch in range(epochs):
        total_loss = 0.0
        for i, subgraph in enumerate(subgraphs, 1):
            subgraph = subgraph.to(device)
            
            optimizer.zero_grad()
            
            with autocast():  # Enable mixed precision
                embeddings, _, _ = model(subgraph.x, subgraph.edge_index)
                mask = torch.randint(0, 2, embeddings.size(), device=device).float()
                loss = criterion(masking_task(embeddings, mask), embeddings)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Clean up memory
            del subgraph, embeddings, mask
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(subgraphs)
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def run_model(model, embeddings, edge_index, logits_index):
    """
    Run forward pass on the model to get logits.

    Args:
        model (nn.Module): The model
        embeddings (Tensor): Node embeddings
        edge_index (Tensor): Edge indices
        logits_index (int): Indices for logits computation

    Returns:
        Tensor: Model output logits
    """
    embeddings, logits, attn_weights = model(embeddings, edge_index)
    return embeddings, logits, attn_weights

def get_high_confidence_predictions(logits, threshold):
    """
    Get high confidence predictions based on threshold.

    Args:
        logits (Tensor): Model output logits
        threshold (float): Confidence threshold

    Returns:
        Tuple[Tensor, Tensor]: High confidence mask and predicted classes
    """
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    high_confidence_mask = max_probs >= threshold
    return high_confidence_mask, preds

def collect_pseudo_labels(indices, preds, pseudo_labels):
    """
    Collect pseudo labels for unlabeled nodes.

    Args:
        indices (numpy.ndarray): Node indices
        preds (numpy.ndarray): Predicted classes
        pseudo_labels (dict): Dictionary of pseudo labels
    """
    for idx, pred in zip(indices, preds):
        pseudo_labels[idx] = pred

def assign_pseudo_labels(model, data, threshold=0.9, device=None):
    """
    Assign pseudo labels to unlabeled samples using batch processing to avoid memory overflow.
    
    Args:
        model (nn.Module): Trained model
        data (Data): PyG data object
        threshold (float): Confidence threshold
        device (torch.device): Computation device
    
    Returns:
        tuple:
            dict: Mapping from node indices to pseudo labels
            torch.Tensor: Updated unknown_mask
    """
    model.eval()
    pseudo_labels = {}
    
    with torch.no_grad():
        # Extract indices of nodes with unknown labels
        unknown_indices = torch.where(data.unknown_mask)[0].cpu().numpy()
        logging.info(f"[assign_pseudo_labels] Number of nodes with unknown labels: {len(unknown_indices)}")
    
        if len(unknown_indices) == 0:
            return pseudo_labels, data.unknown_mask
        
        # Ensure edge_index is on CPU
        edge_index_cpu = data.edge_index.cpu()
    
        subset_size = 1000  # Adjust based on GPU memory
        for i in range(0, len(unknown_indices), subset_size):
            subset = unknown_indices[i:i+subset_size]
            
            try:
                # Create tensor and extract subgraph on CPU
                subset_tensor = torch.tensor(subset, dtype=torch.long)
                
                # Extract subgraph on CPU
                subset_edge_index, _ = subgraph(
                    subset_tensor,
                    edge_index_cpu,
                    relabel_nodes=True,
                    num_nodes=data.num_nodes
                )
                
                # Move necessary data to specified device
                subset_edge_index = subset_edge_index.to(device)
                subset_x = data.x[subset].to(device)
                
                # Run model
                _, logits, _ = run_model(model, subset_x, subset_edge_index, logits_index=1)
        
                # Get high confidence predictions
                high_confidence_mask, preds = get_high_confidence_predictions(logits, threshold)
        
                # Move predictions back to CPU for processing
                high_confidence_mask = high_confidence_mask.cpu()
                preds = preds.cpu()
        
                # Map high confidence nodes back to original graph indices
                high_confidence_indices = subset[high_confidence_mask.numpy()]
                high_confidence_preds = preds[high_confidence_mask].numpy()
        
                # Collect pseudo labels
                collect_pseudo_labels(high_confidence_indices, high_confidence_preds, pseudo_labels)
        
            except Exception as e:
                logging.error(f"[assign_pseudo_labels] Error processing batch {i}: {e}")
                continue
            
            finally:
                # Release memory
                if 'subset_x' in locals():
                    del subset_x
                if 'subset_edge_index' in locals():
                    del subset_edge_index
                if 'logits' in locals():
                    del logits
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    # Update unknown_mask, removing nodes with assigned pseudo labels
    if pseudo_labels:
        assigned_indices = list(pseudo_labels.keys())
        data.unknown_mask[assigned_indices] = False
        logging.info(f"[assign_pseudo_labels] Number of nodes with assigned pseudo labels: {len(assigned_indices)}")
    
    return pseudo_labels, data.unknown_mask

def save_model(model, path):
    """
    Save model state.

    Args:
        model (nn.Module): Model to save
        path (str): Save path
    """
    torch.save(model.state_dict(), path)

def evaluate(model, data, device, batch_size=512):
    """
    Evaluate model accuracy on test set using batch processing to avoid memory overflow.

    Args:
        model (nn.Module): Model to evaluate
        data (Data): PyG data object
        device (torch.device): Computation device
        batch_size (int): Batch size for evaluation

    Returns:
        float: Accuracy score
    """
    model.eval()
    correct = 0
    total = 0

    # Get indices of test nodes
    test_indices = torch.where(data.test_mask)[0].cpu().numpy()
    num_test = len(test_indices)

    if num_test == 0:
        logging.warning("[evaluate] Test set is empty.")
        return 0.0

    # Ensure edge_index is on CPU
    edge_index_cpu = data.edge_index.cpu()
    
    # Process test nodes in batches
    for i in range(0, num_test, batch_size):
        batch_indices = test_indices[i:i+batch_size]
        batch_nodes = list(set(batch_indices))
        
        # Create tensor on CPU
        batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long)

        try:
            # Extract subgraph on CPU
            subset_edge_index, _ = subgraph(
                batch_nodes_tensor,
                edge_index_cpu,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            
            # Move necessary data to specified device
            subset_edge_index = subset_edge_index.to(device)
            subset_x = data.x[batch_nodes].to(device)

            with torch.no_grad():
                # Run model on subgraph
                embeddings, logits, _ = run_model(model, subset_x, subset_edge_index, logits_index=1)

                # Get relative indices
                relative_indices = [batch_nodes.index(idx) for idx in batch_indices]

                # Extract corresponding logits
                logits_batch = logits[relative_indices]

                # Compute predicted classes
                preds = logits_batch.argmax(dim=1)

                # Get true labels and move to correct device
                labels_batch = data.y[batch_indices].to(device)

                # Compute correct predictions
                correct += (preds == labels_batch).sum().item()
                total += len(batch_indices)

        except Exception as e:
            logging.error(f"[evaluate] Error processing batch {i}: {e}")
            logging.error(f"Batch nodes device: CPU")
            logging.error(f"Edge index device before subgraph: CPU")
            logging.error(f"Data edge index device: {data.edge_index.device}")
            continue

        # Release memory
        del subset_x, subset_edge_index, embeddings, logits, preds, labels_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Log and print results
    logging.info(f"[evaluate] Correct predictions: {correct}, Total: {total}, Accuracy: {accuracy:.4f}")

    return accuracy