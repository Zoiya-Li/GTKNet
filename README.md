# GTKNet: Gated Temporal Knowledge Network

GTKNet is a novel deep learning framework for transaction classification that combines Graph Neural Networks (GNNs) with Knowledge Graph embeddings. The model employs a unique fusion mechanism to integrate structural information from transaction graphs with semantic information from knowledge graphs.

## Key Features

- **Multi-source Information Fusion**: Combines transaction graph structure with knowledge graph embeddings
- **Flexible Graph Convolution**: Supports multiple GNN architectures (GraphSAGE, GAT, GraphConv)
- **Self-supervised Learning**: Includes node masking and contrastive learning for better representations
- **Memory-efficient Processing**: Implements batch processing for handling large-scale graphs
- **Comprehensive Loss Function**: Combines classification and triplet losses for enhanced learning

## Project Structure

```
.
├── main.py              # Step 1: Generate graph transformer embeddings
├── model.py            # Core model architecture and loss functions
├── data_processing.py  # Data loading and preprocessing utilities
├── training.py         # Training and evaluation functions
├── fusion.py          # Step 2: Fusion mechanism and evaluation
└── requirements.txt    # Project dependencies
```

## Running the Project

The project runs in two main steps:

### Step 1: Generate Embeddings

First, run `main.py` to generate both types of embeddings:
```bash
python main.py
```
This will:
1. Load and process the transaction graph data
2. Train the graph transformer model
3. Generate and save node embeddings
4. Generate knowledge graph embeddings

Output files:
- `node_embeddings_with_labels.csv`: Graph transformer embeddings
- `KG_node_embeddings.csv`: Knowledge graph embeddings

### Step 2: Fusion and Evaluation

After generating both embeddings, run `fusion.py` to combine them and evaluate the results:
```bash
python fusion.py
```
This will:
1. Load both types of embeddings
2. Apply the gating fusion mechanism
3. Train and evaluate the final model
4. Output performance metrics

## Model Architecture

### Core Components

1. **Graph Transformer Model (GTModel)**
   - Multi-layer graph convolution with configurable architecture
   - Multi-head attention mechanism
   - Dropout and normalization for regularization

2. **Gating Fusion Mechanism**
   - Adaptive integration of graph and knowledge embeddings
   - Learnable gating parameters
   - Batch normalization for stable training

## Requirements

```
torch>=1.8.0
torch-geometric>=2.0.0
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
```

## Model Parameters

- `input_dim`: Dimension of input node features
- `hidden_dim`: Dimension of hidden layers
- `num_layers`: Number of graph convolution layers
- `dropout`: Dropout rate (default: 0.3)
- `num_heads`: Number of attention heads (default: 4)
- `graph_conv_type`: Type of graph convolution ('sage', 'gat', 'graphconv')