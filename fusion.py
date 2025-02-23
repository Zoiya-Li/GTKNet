import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
node_label_file = "./elliptic_bitcoin_dataset/node_embeddings_with_labels.csv"
kg_file = "./elliptic_bitcoin_dataset/KG_node_embeddings.csv"

# 1. Read and align data
df_label = pd.read_csv(node_label_file)
df_kg = pd.read_csv(kg_file)

# Check if 'txId' column exists
if 'txId' not in df_label.columns:
    raise ValueError(f"'txId' column not found in {node_label_file}")
if 'txId' not in df_kg.columns:
    raise ValueError(f"'txId' column not found in {kg_file}")

# Find common transaction IDs
set_label = set(df_label['txId'].astype(str))
set_kg = set(df_kg['txId'].astype(str))
common_txids = set_label.intersection(set_kg)

# Filter data
df_label = df_label[df_label['txId'].astype(str).isin(common_txids)].reset_index(drop=True)
df_kg = df_kg[df_kg['txId'].astype(str).isin(common_txids)].reset_index(drop=True)

# Sort by transaction ID
df_label.sort_values('txId', inplace=True)
df_label.reset_index(drop=True, inplace=True)
df_kg.sort_values('txId', inplace=True)
df_kg.reset_index(drop=True, inplace=True)

# Check alignment
if not all(df_label['txId'].astype(str) == df_kg['txId'].astype(str)):
    raise ValueError("Transaction IDs do not match row by row")

# Remove nodes with label 2 (unknown)
df_filtered = df_label[df_label['label'] != 2].reset_index(drop=True)
keep_txids = set(df_filtered['txId'].astype(str))
df_kg_filtered = df_kg[df_kg['txId'].astype(str).isin(keep_txids)].reset_index(drop=True)

# Sort by transaction ID
df_filtered.sort_values('txId', inplace=True)
df_filtered.reset_index(drop=True, inplace=True)
df_kg_filtered.sort_values('txId', inplace=True)
df_kg_filtered.reset_index(drop=True, inplace=True)

# Check alignment after filtering
if not all(df_filtered['txId'].astype(str) == df_kg_filtered['txId'].astype(str)):
    raise ValueError("Transaction IDs do not match row by row after filtering")

# Extract labels and features
labels = df_filtered['label'].values
gt_embedding_cols = [col for col in df_filtered.columns if col.startswith('embedding_')]
X_gt = df_filtered[gt_embedding_cols].values

kg_embedding_cols = [col for col in df_kg_filtered.columns if col.startswith('embedding_') or col.startswith('KG_emb_')]
X_kg = df_kg_filtered[kg_embedding_cols].values

# Check for NaN or Inf values in input features
if np.isnan(X_gt).any() or np.isnan(X_kg).any():
    raise ValueError("Input features contain NaN values")
if np.isinf(X_gt).any() or np.isinf(X_kg).any():
    raise ValueError("Input features contain Inf values")

# Check feature dimensions
if X_gt.shape[0] != X_kg.shape[0]:
    raise ValueError("Feature dimensions do not match")

# Check label uniqueness
unique_labels = np.unique(labels)
print(f"Unique labels: {unique_labels}")
if not set(unique_labels).issubset({0, 1}):
    raise ValueError(f"Labels contain unexpected classes: {unique_labels}")

# Standardize features
scaler_gt = StandardScaler()
X_gt = scaler_gt.fit_transform(X_gt)

scaler_kg = StandardScaler()
X_kg = scaler_kg.fit_transform(X_kg)

# Convert to tensors
X_gt = torch.tensor(X_gt, dtype=torch.float32)
X_kg = torch.tensor(X_kg, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Define feature dimensions
dim_gt = X_gt.shape[1]
dim_kg = X_kg.shape[1]
if dim_gt != dim_kg:
    raise ValueError(f"Feature dimensions do not match: dim_gt={dim_gt}, dim_kg={dim_kg}")

print("Data standardized, feature statistics:")
print(f"X_gt - min: {X_gt.min().item()}, max: {X_gt.max().item()}, mean: {X_gt.mean().item()}, std: {X_gt.std().item()}")
print(f"X_kg - min: {X_kg.min().item()}, max: {X_kg.max().item()}, mean: {X_kg.mean().item()}, std: {X_kg.std().item()}")

class GatingFusion(nn.Module):
    """
    Gating fusion mechanism for combining graph and knowledge graph embeddings.
    
    Args:
        dim_gt (int): Dimension of graph transformer embeddings
        dim_kg (int): Dimension of knowledge graph embeddings
        fused_dim (int): Dimension of fused output
    """
    def __init__(self, dim_gt, dim_kg, fused_dim=32):
        super(GatingFusion, self).__init__()
        self.linear_g = nn.Linear(dim_gt + dim_kg, 1)
        self.proj = nn.Sequential(
            nn.Linear(dim_gt, fused_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(fused_dim, fused_dim)
        )

    def forward(self, x_gt, x_kg):
        combined = torch.cat([x_gt, x_kg], dim=1)
        g = torch.sigmoid(self.linear_g(combined))
        fused = g * x_gt + (1 - g) * x_kg
        z = self.proj(fused)
        return z

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similar/dissimilar pairs.
    
    Args:
        margin (float): Margin for contrastive loss
        epsilon (float): Small value to prevent numerical instability
    """
    def __init__(self, margin=1.0, epsilon=1e-10):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, z1, z2, label):
        dist = (z1 - z2).pow(2).sum(1).sqrt() + self.epsilon
        pos_loss = label * dist.pow(2)
        neg_loss = (1 - label) * F.relu(self.margin - dist).pow(2)
        loss = (pos_loss + neg_loss).mean() / 2.0
        return loss

def create_pairs(labels, num_pairs=1000):
    """
    Create positive and negative pairs for contrastive learning.
    
    Args:
        labels (array-like): Labels for each sample
        num_pairs (int): Number of pairs to generate
        
    Returns:
        tuple: Arrays of indices for pairs and corresponding labels
    """
    if isinstance(labels, torch.Tensor):
        idx_0 = (labels == 0).nonzero(as_tuple=True)[0]
        idx_1 = (labels == 1).nonzero(as_tuple=True)[0]
    elif isinstance(labels, np.ndarray):
        idx_0 = np.nonzero(labels == 0)[0]
        idx_1 = np.nonzero(labels == 1)[0]
    else:
        raise TypeError("Unsupported type for labels")

    if len(idx_0) < 2 or len(idx_1) < 2:
        raise ValueError("Not enough classes to create pairs")

    pairs = []
    half = num_pairs // 2
    for _ in range(half):
        if torch.rand(1).item() > 0.5 and len(idx_0) > 1:
            i, j = torch.randint(0, len(idx_0), (2,))
            pairs.append((idx_0[i].item(), idx_0[j].item(), 1))
        else:
            if len(idx_1) > 1:
                i, j = torch.randint(0, len(idx_1), (2,))
                pairs.append((idx_1[i].item(), idx_1[j].item(), 1))

    for _ in range(num_pairs - len(pairs)):
        if len(idx_0) > 0 and len(idx_1) > 0:
            i = torch.randint(0, len(idx_0), (1,)).item()
            j = torch.randint(0, len(idx_1), (1,)).item()
            pairs.append((idx_0[i].item(), idx_1[j].item(), 0))

    # Check generated pairs
    for pair in pairs:
        i, j, label = pair
        if i >= len(labels) or j >= len(labels):
            raise ValueError("Generated indices out of range")
        if label not in [0, 1]:
            raise ValueError("Generated label is not 0 or 1")
        if label == 1 and labels[i] != labels[j]:
            raise ValueError("Similar pair label is 1 but classes do not match")
        if label == 0 and labels[i] == labels[j]:
            raise ValueError("Dissimilar pair label is 0 but classes match")

    return pairs

def initialize_weights(m):
    """Initialize network weights using Xavier initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Set number of experiment repetitions
num_experiments = 1  # Adjust as needed

# Define training proportions
training_proportions = [0.7]
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
all_results = {p: {metric: [] for metric in metrics} for p in training_proportions}

for train_size in training_proportions:
    print(f"\n===== Training proportion: {train_size} =====")
    for experiment in range(1, num_experiments + 1):
        print(f"\nStarting experiment {experiment}/{num_experiments} (train_ratio={train_size})")

        # Set random seed for reproducibility
        seed = int(80 + 2 * experiment + train_size * 201)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Split data into training and testing sets
        X_train_gt, X_test_gt, X_train_kg, X_test_kg, y_train, y_test = train_test_split(
            X_gt.numpy(), X_kg.numpy(), labels.numpy(), train_size=train_size, random_state=seed, stratify=labels.numpy()
        )

        # Convert to tensors and move to device
        X_train_gt_t = torch.tensor(X_train_gt, dtype=torch.float32).to(device)
        X_train_kg_t = torch.tensor(X_train_kg, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

        X_test_gt_t = torch.tensor(X_test_gt, dtype=torch.float32).to(device)
        X_test_kg_t = torch.tensor(X_test_kg, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

        # Dynamically set num_pairs
        num_pairs = min(2000, len(y_train) * 10)  # Adjusted for debugging
        pairs = create_pairs(y_train_t, num_pairs=num_pairs)

        # Initialize model, loss function, and optimizer
        fusion_model = GatingFusion(dim_gt, dim_kg, fused_dim=64).to(device)  # Adjusted hidden layer size
        fusion_model.apply(initialize_weights)  # Initialize weights
        criterion = ContrastiveLoss(margin=10.0, epsilon=1e-10)  # Added epsilon
        optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001)  # Adjusted learning rate

        batch_size = 64  # Adjusted batch size for debugging
        num_epochs = 100  # Adjusted number of epochs for debugging

        for epoch in range(num_epochs):
            fusion_model.train()
            pairs_array = np.array(pairs)
            np.random.shuffle(pairs_array)
            pairs_shuffled = pairs_array.tolist()
            total_loss = 0.0
            for start in range(0, len(pairs_shuffled), batch_size):
                end = start + batch_size
                batch = pairs_shuffled[start:end]
                i_idx = [b[0] for b in batch]
                j_idx = [b[1] for b in batch]
                y_batch = [b[2] for b in batch]

                i_idx = torch.tensor(i_idx, dtype=torch.long, device=device)
                j_idx = torch.tensor(j_idx, dtype=torch.long, device=device)
                y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device)

                z_i = fusion_model(X_train_gt_t[i_idx], X_train_kg_t[i_idx])
                z_j = fusion_model(X_train_gt_t[j_idx], X_train_kg_t[j_idx])

                # Print z_i and z_j statistics
                # print(f"Epoch {epoch+1}, Batch {start//batch_size +1}")
                # print(f"z_i - min: {z_i.min().item()}, max: {z_i.max().item()}, mean: {z_i.mean().item()}, std: {z_i.std().item()}")
                # print(f"z_j - min: {z_j.min().item()}, max: {z_j.max().item()}, mean: {z_j.mean().item()}, std: {z_j.std().item()}")

                # Check z_i and z_j for NaN or Inf values
                if torch.isnan(z_i).any() or torch.isnan(z_j).any():
                    print(f"NaN detected in model outputs at epoch {epoch+1}, batch {start//batch_size +1}")
                    raise ValueError("Model outputs contain NaN")
                if torch.isinf(z_i).any() or torch.isinf(z_j).any():
                    print(f"Inf detected in model outputs at epoch {epoch+1}, batch {start//batch_size +1}")
                    raise ValueError("Model outputs contain Inf")

                loss = criterion(z_i, z_j, y_batch)

                # Check loss for NaN or Inf values
                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at epoch {epoch+1}, batch {start//batch_size +1}")
                    raise ValueError("Loss function result is NaN")
                if torch.isinf(loss).any():
                    print(f"Inf detected in loss at epoch {epoch+1}, batch {start//batch_size +1}")
                    raise ValueError("Loss function result is Inf")

                optimizer.zero_grad()
                loss.backward()

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item() * len(batch)

            avg_loss = total_loss / len(pairs_shuffled)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Extract fused features after contrastive learning
        fusion_model.eval()
        with torch.no_grad():
            fused_train = fusion_model(X_train_gt_t, X_train_kg_t).cpu().numpy()
            fused_test = fusion_model(X_test_gt_t, X_test_kg_t).cpu().numpy()
        y_train_np = y_train_t.cpu().numpy()
        y_test_np = y_test_t.cpu().numpy()

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(fused_train, y_train_np)
        y_pred = clf.predict(fused_test)
        y_proba = clf.predict_proba(fused_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test_np, y_pred)
        prec = precision_score(y_test_np, y_pred, zero_division=0)
        rec = recall_score(y_test_np, y_pred, zero_division=0)
        f1 = f1_score(y_test_np, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_np, y_proba)

        print(f"Experiment {experiment} results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # Record metrics
        all_results[train_size]['accuracy'].append(acc)
        all_results[train_size]['precision'].append(prec)
        all_results[train_size]['recall'].append(rec)
        all_results[train_size]['f1'].append(f1)
        all_results[train_size]['auc'].append(auc)

# Calculate mean and standard deviation of metrics
mean_results = {metric: [] for metric in metrics}
std_results = {metric: [] for metric in metrics}

# Print mean and standard deviation of metrics for all experiments
print("\nMean and standard deviation of metrics for all experiments:")
for p in training_proportions:
    print(f"\nTraining proportion: {p}")
    for metric in metrics:
        mean = np.mean(all_results[p][metric])
        std = np.std(all_results[p][metric])
        print(f"{metric.capitalize()}: Mean = {mean:.4f}")#, Standard Deviation = {std**2:.6f}")