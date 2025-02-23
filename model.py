"""
Model definitions for the GTKNet architecture.
This module contains the main model classes including the Graph Transformer model,
node masking task, and comprehensive loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GraphConv
from torch.nn import MultiheadAttention

class GTModel(nn.Module):
    """
    Graph Transformer Model for transaction classification.
    Combines graph convolution with multi-head attention for enhanced feature learning.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, 
                 graph_conv_type='sage', num_heads=4, num_classes=3):
        """
        Initialize the Graph Transformer model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            num_layers (int): Number of graph convolution layers
            dropout (float): Dropout probability
            graph_conv_type (str): Graph convolution type ('sage', 'gat', 'graphconv')
            num_heads (int): Number of attention heads (only for GAT)
            num_classes (int): Number of classes for classification
        """
        super(GTModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Select graph convolution layer type
        if graph_conv_type.lower() == 'sage':
            conv_layer = SAGEConv
            conv_args = {'in_channels': -1, 'out_channels': -1}  # Use standard parameter names
        elif graph_conv_type.lower() == 'gat':
            conv_layer = GATConv
            conv_args = {'heads': num_heads, 'dropout': dropout}
        else:
            conv_layer = GraphConv
            conv_args = {}
        
        self.graph_convs = nn.ModuleList()
        for i in range(num_layers):
            if graph_conv_type.lower() == 'sage':
                conv_args['in_channels'] = input_dim if i == 0 else hidden_dim
                conv_args['out_channels'] = hidden_dim
                self.graph_convs.append(conv_layer(**conv_args))
            else:
                self.graph_convs.append(
                    conv_layer(input_dim if i == 0 else hidden_dim, hidden_dim, **conv_args)
                )
        
        # Multi-head attention mechanism
        self.attention = MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Gated mechanism
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        self.cell_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)  # Modified to num_classes
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix of shape [N, input_dim]
            edge_index (Tensor): Edge index of shape [2, E]
            batch (Tensor, optional): Batch information of shape [N]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (embeddings, logits, attention_weights)
        """
        # Graph convolution layers
        graph_features = x
        for i, conv in enumerate(self.graph_convs):
            new_features = conv(graph_features, edge_index)
            new_features = self.act(new_features)
            new_features = F.dropout(new_features, p=self.dropout, training=self.training)
            if i > 0:
                graph_features = graph_features + new_features
            else:
                graph_features = new_features
        
        # Multi-head attention mechanism
        graph_features_attn = graph_features.unsqueeze(0)  # (1, N, hidden_dim)
        attended_features, attn_weights = self.attention(graph_features_attn, graph_features_attn, graph_features_attn)
        attended_features = attended_features.squeeze(0)  # (N, hidden_dim)
        
        # Gated mechanism
        combined = torch.cat([graph_features, attended_features], dim=1)
        f = self.forget_gate(combined)
        i = self.input_gate(combined)
        c_tilde = self.cell_gate(combined)
        cell_state = f * graph_features + i * c_tilde
        
        # LayerNorm
        cell_state = self.layer_norm(cell_state)
        
        # Embedding
        embeddings = self.embedding_layer(cell_state)
        
        # Classification result
        logits = self.classifier(embeddings)
        
        return embeddings, logits, attn_weights


class NodeMaskingTask(nn.Module):
    """
    Node masking task for self-supervised learning.
    Used for reconstruction tasks in the pre-training phase.
    """
    
    def __init__(self, embedding_dim, hidden_dim=64):
        """
        Initialize the node masking task.

        Args:
            embedding_dim (int): Node embedding dimension
            hidden_dim (int): Hidden layer dimension
        """
        super(NodeMaskingTask, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim // 2)
        )
    
    def forward(self, embeddings, mask):
        """
        Forward pass for node reconstruction.

        Args:
            embeddings (Tensor): Node embeddings of shape [N, embedding_dim]
            mask (Tensor): Mask tensor of shape [N], 1 indicates masked nodes

        Returns:
            Tensor: Reconstructed node embeddings of shape [N, embedding_dim]
        """
        masked_embeddings = embeddings.clone()
        masked_embeddings[mask == 1] = 0  # Masked part of the embeddings
        reconstructed = self.encoder(masked_embeddings)
        return reconstructed


class ComprehensiveLoss(nn.Module):
    """
    Comprehensive loss function combining classification and triplet losses.
    Used for semi-supervised learning with contrastive objectives.
    """
    
    def __init__(self, triplet_margin=1.0):
        """
        Initialize the comprehensive loss function.

        Args:
            triplet_margin (float): Margin for triplet loss
        """
        super(ComprehensiveLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin, p=2)
    
    def forward(self, logits_anchor, labels_anchor,
                logits_positive, labels_positive,
                logits_negative, labels_negative,
                embeddings_anchor, embeddings_positive, embeddings_negative):
        """
        Forward pass for loss computation.

        Args:
            logits_anchor (Tensor): Classification output for anchor nodes [B, num_classes]
            labels_anchor (Tensor): True labels for anchor nodes [B]
            logits_positive (Tensor): Classification output for positive nodes [B, num_classes]
            labels_positive (Tensor): True labels for positive nodes [B]
            logits_negative (Tensor): Classification output for negative nodes [B, num_classes]
            labels_negative (Tensor): True labels for negative nodes [B]
            embeddings_anchor (Tensor): Embeddings for anchor nodes [B, embedding_dim]
            embeddings_positive (Tensor): Embeddings for positive nodes [B, embedding_dim]
            embeddings_negative (Tensor): Embeddings for negative nodes [B, embedding_dim]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (total_loss, classification_loss, triplet_loss)
        """
        # Classification loss
        loss_cls_anchor = F.cross_entropy(logits_anchor, labels_anchor)
        loss_cls_positive = F.cross_entropy(logits_positive, labels_positive)
        loss_cls_negative = F.cross_entropy(logits_negative, labels_negative)
        loss_cls = (loss_cls_anchor + loss_cls_positive + loss_cls_negative) / 3.0
        
        # Triplet loss
        loss_triplet = self.triplet_loss(embeddings_anchor, embeddings_positive, embeddings_negative)
        
        # Comprehensive loss
        loss = loss_cls + loss_triplet
        return loss, loss_cls, loss_triplet