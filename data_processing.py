"""
Data processing utilities for the GTKNet model.
This module handles data loading, preprocessing, and dataset creation for the graph neural network.
"""

import pandas as pd
import numpy as np
import torch
import networkx as nx
import os
import logging
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from torch_geometric.data import Data
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(feature_path, label_path, edgelist_path):
    """
    Load features, labels, and edge list data to construct a graph data object.

    Args:
        feature_path (str): Path to feature file
        label_path (str): Path to label file
        edgelist_path (str): Path to edge list file

    Returns:
        Data: PyG data object
        dict: Mapping from node IDs to indices
    """
    # Load feature file
    features_df = pd.read_csv(feature_path)
    features = torch.tensor(features_df.drop(columns=['txId']).values, dtype=torch.float)
    node_ids = features_df['txId'].astype(str).values  # Ensure string type

    # Create node to index mapping
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Load edge list file
    edgelist_df = pd.read_csv(edgelist_path)
    
    # Keep only nodes that are defined in the feature file
    edgelist_df = edgelist_df[edgelist_df['txId1'].astype(str).isin(node_to_idx.keys()) &
                             edgelist_df['txId2'].astype(str).isin(node_to_idx.keys())]

    # Convert to numpy array first, then create tensor
    edge_index_array = np.array([
        edgelist_df['txId1'].astype(str).map(node_to_idx).values,
        edgelist_df['txId2'].astype(str).map(node_to_idx).values
    ])
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)

    # Load label file
    labels_df = pd.read_csv(label_path)

    # Convert label column to string type
    labels_df['class'] = labels_df['class'].astype(str)

    # Create label mapping
    label_mapping = {'illicit': 0, 'licit': 1, 'unknown': 2}

    # Check if all labels are in the mapping
    missing_classes = set(labels_df['class']) - set(label_mapping.keys())

    # Map labels
    labels_df['class_encoded'] = labels_df['class'].map(label_mapping).fillna(2).astype(int)

    # Initialize label tensor with default label 'unknown' (2)
    labels_tensor = torch.full((len(node_to_idx),), 2, dtype=torch.long)

    # Set known labels
    labels_df = labels_df.set_index('txId')
    existing_labels = 0
    for node_id, encoded_label in zip(labels_df.index, labels_df['class_encoded']):
        node_id_str = str(node_id)  # Ensure txId is string type
        if node_id_str in node_to_idx:
            idx = node_to_idx[node_id_str]
            labels_tensor[idx] = encoded_label
            existing_labels += 1
        else:
            logging.warning(f"Label file contains txId '{node_id_str}' not found in feature file.")

    # Create graph data object
    data = Data(x=features, edge_index=edge_index, y=labels_tensor)

    # Record label mapping results
    num_label_0 = (labels_tensor == 0).sum().item()
    num_label_1 = (labels_tensor == 1).sum().item()
    num_label_unknown = (labels_tensor == 2).sum().item()
    logging.info(f"Number of nodes with label 0: {num_label_0}")
    logging.info(f"Number of nodes with label 1: {num_label_1}")
    logging.info(f"Number of nodes with label unknown: {num_label_unknown}")

    return data, node_to_idx

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train, validation, and test sets. Only splits nodes with labels 0 and 1,
    and creates an unknown_mask for nodes with label 2.

    Args:
        data (Data): PyG data object
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        random_state (int): Random seed

    Returns:
        Data: Updated data object with train_mask, val_mask, test_mask, and unknown_mask
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."
    
    num_nodes = data.num_nodes
    labels = data.y.numpy()
    
    # Create unknown_mask for nodes with label 2
    unknown_mask = labels == 2
    known_mask = ~unknown_mask  # Nodes with labels 0 or 1
    
    known_indices = np.where(known_mask)[0]
    unknown_indices = np.where(unknown_mask)[0]
    
    # Get labels for known nodes
    known_labels = labels[known_indices]
    
    # Split known nodes into train and temporary set (train + validation)
    train_val_indices, test_indices = train_test_split(
        known_indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=known_labels
    )
    
    # Calculate validation set ratio relative to train and validation sets
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    # Split train and validation sets
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=labels[train_val_indices]
    )
    
    # Initialize all masks as False
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Set train, validation, and test masks
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Add unknown_mask to data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.unknown_mask = torch.tensor(unknown_mask, dtype=torch.bool)
    
    # Record information
    num_train = train_mask.sum().item()
    num_val = val_mask.sum().item()
    num_test = test_mask.sum().item()
    num_unknown = unknown_mask.sum()
    
    logging.info(f"Number of nodes in training set: {num_train}")
    logging.info(f"Number of nodes in validation set: {num_val}")
    logging.info(f"Number of nodes in test set: {num_test}")
    
    return data

def build_embeddings_tensor(embeddings_df, node_to_idx, num_nodes):
    """
    Build embeddings tensor from DataFrame.

    Args:
        embeddings_df (pd.DataFrame): Node embeddings DataFrame
        node_to_idx (dict): Node to index mapping
        num_nodes (int): Number of nodes

    Returns:
        Tensor: Node embeddings tensor
    """
    embedding_dim = embeddings_df.shape[1]
    embeddings = torch.zeros((num_nodes, embedding_dim), dtype=torch.float)  # Assume GT and KG embeddings are 64D
    for node_id, idx in node_to_idx.items():
        if node_id in embeddings_df.index:
            embeddings[idx, :embedding_dim] = torch.tensor(embeddings_df.loc[node_id].values, dtype=torch.float)
            embeddings[idx, embedding_dim:] = torch.tensor(embeddings_df.loc[node_id].values, dtype=torch.float)  # Placeholder for KG embeddings
        else:
            embeddings[idx, :embedding_dim] = torch.zeros(embedding_dim)
            embeddings[idx, embedding_dim:] = torch.zeros(embedding_dim)
            logging.warning(f"Node {node_id} not found in embeddings, using zero vector instead.")
    return embeddings

def split_graph_into_subgraphs(data, num_subgraphs=10):
    """
    Split large graph into subgraphs.

    Args:
        data (Data): PyG data object
        num_subgraphs (int): Number of subgraphs

    Returns:
        List[Data]: List of subgraphs
    """
    import torch_geometric.utils as utils

    # Use simple node-based splitting
    num_nodes = data.num_nodes
    nodes_per_subgraph = num_nodes // num_subgraphs
    subgraphs = []

    for i in range(num_subgraphs):
        start = i * nodes_per_subgraph
        end = (i + 1) * nodes_per_subgraph if i != num_subgraphs - 1 else num_nodes
        sub_nodes = torch.arange(start, end, device='cpu')  # Ensure sub_nodes is on CPU

        # Ensure edge_index is on CPU
        sub_edge_index, _ = utils.subgraph(
            sub_nodes,
            data.edge_index.cpu(),  # Ensure edge_index is on CPU
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        sub_x = data.x[start:end]
        sub_y = data.y[start:end]

        # Create subgraph data object
        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
        subgraphs.append(sub_data)

    return subgraphs

def prepare_embedding(G, data, node_to_idx, device, embeddings_file, model_dir):
    """
    Prepare and save node embeddings using Node2Vec.

    Args:
        G (dict): Graph adjacency list representation
        data (Data): PyG data object
        node_to_idx (dict): Node to index mapping
        device (torch.device): Computation device
        embeddings_file (str): Path to save embeddings
        model_dir (str): Model directory

    Returns:
        pd.DataFrame: Node embeddings DataFrame
    """

    # Define Node2Vec model file path
    model_path = os.path.join(model_dir, 'Node_KG.pkl')

    # Create reverse mapping
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    # Extract edge list and use original transaction IDs
    edges = G['edge_index'].cpu().numpy().T
    edges = [(idx_to_node[src], idx_to_node[dst]) for src, dst in edges]

    # Create networkx graph
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)

    if os.path.exists(model_path):
        # Load existing Node2Vec model (KeyedVectors)
        model = KeyedVectors.load(model_path, mmap='r')
        logging.info(f"Loaded existing Node2Vec model: {model_path}")
    else:
        # Initialize Node2Vec
        node2vec = Node2Vec(
            G_nx,
            dimensions=16,
            walk_length=50,
            num_walks=70,
            p=0.5,
            q=2.0,
            workers=4
        )

        # Train Node2Vec model
        node2vec_model = node2vec.fit(window=5, min_count=1, batch_words=4)

        # Save Node2Vec model's KeyedVectors part
        node2vec_model.wv.save(model_path)
        logging.info(f"Trained and saved Node2Vec model to: {model_path}")

        # Use trained model
        model = node2vec_model.wv

    # Extract embeddings - use original transaction IDs
    embeddings = []
    missing_nodes = 0
    node_ids = list(node_to_idx.keys())
    for node_id in node_ids:  # Use original transaction IDs
        if node_id in model:
            embeddings.append(model[node_id])
        else:
            embeddings.append(np.zeros(model.vector_size))
            missing_nodes += 1
            logging.warning(f"Node {node_id} not found in Node2Vec model, using zero vector instead.")
    
    if missing_nodes > 0:
        logging.warning(f"Total {missing_nodes} nodes not found in embeddings")

    # Create DataFrame with embeddings
    embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(len(embeddings[0]))])
    embeddings_df.insert(0, 'txId', node_ids)  # Add txId as first column
    embeddings_df.to_csv(embeddings_file, index=False)

    return embeddings_df

class TripletDataset(Dataset):
    """
    Dataset class for triplet loss training.
    Generates triplets (anchor, positive, negative) for contrastive learning.
    """
    
    def __init__(self, labels):
        """
        Initialize TripletDataset.

        Args:
            labels (List[int]): List of node labels
        """
        self.labels = labels
        self.label_to_indices = self._map_labels_to_indices()
        self.triplets = []
        self.generate_initial_triplets()
        logging.info(f"[TripletDataset] Initialized with {len(self.triplets)} triplets.")

    def _map_labels_to_indices(self):
        """
        Create mapping from labels to node indices.

        Returns:
            dict: Label to node indices mapping
        """
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices
    
    def generate_initial_triplets(self):
        """Generate initial set of triplets."""
        self.triplets = self._generate_triplets()
    
    def _generate_triplets(self, new_labels=None):
        """
        Generate triplets (anchor, positive, negative).

        Args:
            new_labels (List[Tuple[int, int]]): List of newly assigned labels as (node_idx, label)

        Returns:
            List[Tuple[int, int, int]]: List of triplets
        """
        triplets = []
        if new_labels is None:
            # Generate all triplets
            for anchor_idx, anchor_label in enumerate(self.labels):
                if anchor_label == 2:
                    continue  # Skip nodes with label 2
                
                if anchor_label not in self.label_to_indices or len(self.label_to_indices[anchor_label]) < 2:
                    continue  # Not enough positive samples
                # Choose a different positive sample
                positive_indices = self.label_to_indices[anchor_label].copy()
                positive_indices.remove(anchor_idx)
                if not positive_indices:
                    continue  # No positive samples left
                positive_idx = np.random.choice(positive_indices)
                
                # Choose a negative sample
                negative_labels = [l for l in self.label_to_indices.keys() if l != anchor_label]
                if not negative_labels:
                    continue  # No negative samples
                negative_label = np.random.choice(negative_labels)
                negative_idx = np.random.choice(self.label_to_indices[negative_label])
                
                triplets.append((anchor_idx, positive_idx, negative_idx))
        else:
            # Generate triplets only for new labels
            for node_idx, label in new_labels:
                if label == 2:
                    continue  # Skip nodes with label 2
    
                if label not in self.label_to_indices or len(self.label_to_indices[label]) < 2:
                    continue  # Not enough positive samples
                # Choose a different positive sample
                positive_indices = self.label_to_indices[label].copy()
                positive_indices.remove(node_idx)
                if not positive_indices:
                    continue  # No positive samples left
                positive_idx = np.random.choice(positive_indices)
                
                # Choose a negative sample
                negative_labels = [l for l in self.label_to_indices.keys() if l != label]
                if not negative_labels:
                    continue  # No negative samples
                negative_label = np.random.choice(negative_labels)
                negative_idx = np.random.choice(self.label_to_indices[negative_label])
                
                triplets.append((node_idx, positive_idx, negative_idx))
        
        return triplets
    
    def add_triplets_from_new_labels(self, new_labels):
        """
        Add triplets based on newly assigned labels.

        Args:
            new_labels (List[Tuple[int, int]]): List of newly assigned labels as (node_idx, label)
        """
        # Update label to indices mapping
        for node_idx, label in new_labels:
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(node_idx)
        
        # Generate new triplets
        new_triplets = self._generate_triplets(new_labels=new_labels)
        self.triplets.extend(new_triplets)
        logging.info(f"[TripletDataset] Added {len(new_triplets)} new triplets, total: {len(self.triplets)}")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        # Add assertions to ensure index validity
        assert anchor < len(self.labels), f"Anchor index {anchor} out of range"
        assert positive < len(self.labels), f"Positive index {positive} out of range"
        assert negative < len(self.labels), f"Negative index {negative} out of range"
        return (anchor, positive, negative)