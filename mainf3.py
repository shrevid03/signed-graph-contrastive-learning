import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import dgl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# --- High-Performance Configuration ---
class Config:
    DATA_PATH = 'bitcoin-otc.csv'
    EPOCHS = 80
    LR = 0.005
    EMBEDDING_DIM = 128
    NUM_HEADS = 8
    TAU = 0.05
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.5
    BATCH_SIZE = 2048
    BETA = 0.1
    ALPHA = 0.5
    CENTRALITY_TOP_K_RATIO = 0.1

    NUM_WORKERS = 0
    TEST_SPLIT_RATIO = 0.15

# --- Data Loading and Graph Construction ---
def load_and_split_data(path, test_ratio):
    print(f"Loading data from '{path}' and splitting into train/test sets...")
    df = pd.read_csv(path, sep=',', header=None, comment='#', names=['source', 'target', 'rating', 'time'])
    unique_nodes = pd.unique(df[['source', 'target']].values.ravel())
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    df['source'], df['target'] = df['source'].map(node_map), df['target'].map(node_map)
    df['label'] = (df['rating'] > 0).astype(int)
    all_edges, all_labels = df[['source', 'target']].values.tolist(), df['label'].values.tolist()
    num_nodes = len(node_map)

    train_edges, test_edges, train_labels, test_labels = train_test_split(
        all_edges, all_labels, test_size=test_ratio, random_state=42, stratify=all_labels)

    train_pos_edges = [edge for i, edge in enumerate(train_edges) if train_labels[i] == 1]
    train_neg_edges = [edge for i, edge in enumerate(train_edges) if train_labels[i] == 0]
    pos_src, pos_dst = zip(*train_pos_edges) if train_pos_edges else ([], [])
    neg_src, neg_dst = zip(*train_neg_edges) if train_neg_edges else ([], [])

    graph_data = {
        ('node', 'positive', 'node'): (torch.tensor(pos_src), torch.tensor(pos_dst)),
        ('node', 'negative', 'node'): (torch.tensor(neg_src), torch.tensor(neg_dst))
    }
    g_train = dgl.heterograph(graph_data, num_nodes_dict={'node': num_nodes})
    g_train.ndata['feat'] = torch.randn(num_nodes, Config.EMBEDDING_DIM)
    
    print(f"Training Graph: {g_train.num_nodes()} nodes, {g_train.num_edges('positive')} pos edges, {g_train.num_edges('negative')} neg edges.")
    print(f"Test Set: {len(test_edges)} edges.")
    
    return g_train, torch.tensor(train_edges), torch.tensor(train_labels, dtype=torch.long), torch.tensor(test_edges), torch.tensor(test_labels, dtype=torch.long)

# --- Augmentation Methods ---
def augment_with_random_walk(g):
    num_nodes = g.num_nodes()
    pos_src, pos_dst = g.edges(etype='positive')
    A_p = sp.coo_matrix((np.ones(len(pos_src)), (pos_src.cpu().numpy(), pos_dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    neg_src, neg_dst = g.edges(etype='negative')
    A_n = sp.coo_matrix((np.ones(len(neg_src)), (neg_src.cpu().numpy(), neg_dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    A_signed = A_p.astype(float) - A_n.astype(float)
    S2 = (A_signed @ A_signed).tocoo()
    S2.setdiag(0); S2.eliminate_zeros()
    new_pos_src, new_pos_dst = torch.from_numpy(S2.row[S2.data > 0]), torch.from_numpy(S2.col[S2.data > 0])
    new_neg_src, new_neg_dst = torch.from_numpy(S2.row[S2.data < 0]), torch.from_numpy(S2.col[S2.data < 0])
    g_aug = dgl.heterograph({
        ('node', 'positive', 'node'): (new_pos_src, new_pos_dst),
        ('node', 'negative', 'node'): (new_neg_src, new_neg_dst)
    }, num_nodes_dict={'node': num_nodes})
    return g_aug

def augment_with_centrality(g, top_k_ratio):
    num_nodes = g.num_nodes()
    pos_src, pos_dst = g.edges(etype='positive')
    A_pos = sp.coo_matrix((np.ones(len(pos_src)), (pos_src.cpu().numpy(), pos_dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    neg_src, neg_dst = g.edges(etype='negative')
    A_neg = sp.coo_matrix((np.ones(len(neg_src)), (neg_src.cpu().numpy(), neg_dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    A_signed = A_pos - A_neg

    try:
        _, eigenvectors = eigs(A_signed.asfptype(), k=1, which='LR')
        centrality = np.real(eigenvectors).flatten()
    except:
        centrality = (A_pos - A_neg).sum(axis=1).A.flatten()

    num_top_k = int(num_nodes * top_k_ratio)
    sorted_indices = np.argsort(centrality)
    top_pos_nodes, top_neg_nodes = sorted_indices[-num_top_k:], sorted_indices[:num_top_k]

    num_new_links = num_top_k * 5 
    new_pos_src = np.random.choice(top_pos_nodes, size=num_new_links)
    new_pos_dst = np.random.choice(top_pos_nodes, size=num_new_links)
    new_neg_src = np.random.choice(top_neg_nodes, size=num_new_links)
    new_neg_dst = np.random.choice(np.arange(num_nodes), size=num_new_links)

    g_aug = dgl.heterograph({
        ('node', 'positive', 'node'): (torch.from_numpy(new_pos_src), torch.from_numpy(new_pos_dst)),
        ('node', 'negative', 'node'): (torch.from_numpy(new_neg_src), torch.from_numpy(new_neg_dst))
    }, num_nodes_dict={'node': num_nodes})
    return g_aug

# --- Model Architectures ---
class SGCL_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(SGCL_Encoder, self).__init__()
        self.pos_conv = dgl.nn.pytorch.GATConv(in_dim, hidden_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.neg_conv = dgl.nn.pytorch.GATConv(in_dim, hidden_dim // num_heads, num_heads, allow_zero_in_degree=True)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        g_pos = g.edge_type_subgraph(['positive'])
        g_neg = g.edge_type_subgraph(['negative'])
        num_nodes = max(g_pos.num_nodes(), g_neg.num_nodes(), g.num_nodes(), 1)
        h_pos = torch.zeros(num_nodes, self.pos_conv._out_feats * self.pos_conv._num_heads, device=features.device)
        if g_pos.num_edges() > 0:
            h_pos[g_pos.nodes()] = self.pos_conv(g_pos, features).flatten(1)
        h_neg = torch.zeros(num_nodes, self.neg_conv._out_feats * self.neg_conv._num_heads, device=features.device)
        if g_neg.num_edges() > 0:
            h_neg[g_neg.nodes()] = self.neg_conv(g_neg, features).flatten(1)
        h_final = self.final_mlp(self.dropout(torch.cat([h_pos, h_neg], dim=1)))
        return h_pos, h_neg, h_final

class ScorePredictor(nn.Module):
    def __init__(self, in_features, dropout):
        super(ScorePredictor, self).__init__()
        self.layer1 = nn.Linear(in_features * 2, in_features)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(in_features, 1)

    def forward(self, emb_u, emb_v):
        h = torch.cat([emb_u, emb_v], dim=1)
        h = self.dropout(F.relu(self.layer1(h)))
        return self.layer2(h)

# --- Loss Functions ---
def inter_view_loss(pos_view1, pos_view2, neg_view1, neg_view2, tau):
    def single_view_loss(view1, view2):
        view1, view2 = F.normalize(view1, p=2, dim=1), F.normalize(view2, p=2, dim=1)
        sim = torch.mm(view1, view2.T) / tau
        return F.cross_entropy(sim, torch.arange(view1.size(0), device=view1.device))
    return single_view_loss(pos_view1, pos_view2) + single_view_loss(neg_view1, neg_view2)

def intra_view_loss(final_emb, pos_emb, neg_emb, tau):
    final_emb, pos_emb, neg_emb = F.normalize(final_emb, p=2, dim=1), F.normalize(pos_emb, p=2, dim=1), F.normalize(neg_emb, p=2, dim=1)
    sim_pos = torch.exp((final_emb * pos_emb).sum(1) / tau)
    sim_neg = torch.exp((final_emb * neg_emb).sum(1) / tau)
    return -torch.log(sim_pos / (sim_pos + sim_neg)).mean()

# --- Plotting and Metrics ---
def evaluate_and_plot(y_true, y_pred_proba, stage="Final Test"):
    y_pred = (y_pred_proba > 0.5).astype(int)
    print(f"\n--- {stage} Evaluation Metrics ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score (Binary): {f1_score(y_true, y_pred, average='binary', zero_division=0):.4f}")
    print(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("----------------------------------\n")

    if stage == "Final Test":
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix'); plt.savefig('confusion_matrix.png'); plt.close()
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_true, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend(loc="lower right"); plt.savefig('roc_curve.png'); plt.close()
        print("Plots saved.")

# --- Main Logic ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(Config.DATA_PATH): print(f"Error: Dataset file '{Config.DATA_PATH}' not found."); return

    g_train, train_edges, train_labels, test_edges, test_labels = load_and_split_data(Config.DATA_PATH, Config.TEST_SPLIT_RATIO)

    encoder = SGCL_Encoder(in_dim=Config.EMBEDDING_DIM, hidden_dim=Config.EMBEDDING_DIM, out_dim=Config.EMBEDDING_DIM, num_heads=Config.NUM_HEADS, dropout=Config.DROPOUT).to(device)
    predictor = ScorePredictor(Config.EMBEDDING_DIM, Config.DROPOUT).to(device)
    
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = Adam(params, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    loss_fn_supervised = nn.BCEWithLogitsLoss()
    
    train_dataset = TensorDataset(train_edges, train_labels)
    dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    print("\n--- Starting Joint Training with Dual Augmentation ---")
    for epoch in range(1, Config.EPOCHS + 1):
        encoder.train()
        predictor.train()
        
        # --- NEW: Create two different augmented graphs once per epoch ---
        print("Creating augmentations for the new epoch...")
        g_aug_rw = augment_with_random_walk(g_train).to(device)
        g_aug_cent = augment_with_centrality(g_train, Config.CENTRALITY_TOP_K_RATIO).to(device)
        g_train_dev = g_train.to(device)
        
        # Share features
        g_train_dev.ndata['feat'] = g_train.ndata['feat'].to(device)
        g_aug_rw.ndata['feat'] = g_train_dev.ndata['feat']
        g_aug_cent.ndata['feat'] = g_train_dev.ndata['feat']
            
        for edges, labels in dataloader:
            u, v, labels = edges[:, 0].to(device), edges[:, 1].to(device), labels.to(device).float()
            
            # 1. Get embeddings from original graph for supervised and intra-view loss
            pos_orig, neg_orig, final_orig = encoder(g_train_dev, g_train_dev.ndata['feat'])
            
            # 2. Get embeddings from the two different augmented graphs for inter-view loss
            pos_aug1, neg_aug1, _ = encoder(g_aug_rw, g_aug_rw.ndata['feat'])
            pos_aug2, neg_aug2, _ = encoder(g_aug_cent, g_aug_cent.ndata['feat'])
            
            # 3. Supervised Loss (on original graph embeddings)
            emb_u, emb_v = final_orig[u], final_orig[v]
            logits = predictor(emb_u, emb_v).squeeze()
            loss_label = loss_fn_supervised(logits, labels)

            # 4. Contrastive Losses
            loss_inter = inter_view_loss(pos_aug1, pos_aug2, neg_aug1, neg_aug2, Config.TAU)
            loss_intra = intra_view_loss(final_orig, pos_orig, neg_orig, Config.TAU)
            
            # 5. Combined Loss
            loss = loss_label + Config.BETA * (loss_inter + Config.ALPHA * loss_intra)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation on test set for this epoch
        encoder.eval(); predictor.eval()
        with torch.no_grad():
            _, _, final_embeddings = encoder(g_train_dev, g_train_dev.ndata['feat'])
            test_u, test_v = test_edges[:, 0], test_edges[:, 1]
            emb_u_test, emb_v_test = final_embeddings[test_u].to(device), final_embeddings[test_v].to(device)
            test_logits = predictor(emb_u_test, emb_v_test).squeeze()
            test_pred_proba = torch.sigmoid(test_logits).cpu().numpy()
            test_accuracy = accuracy_score(test_labels.numpy(), (test_pred_proba > 0.5).astype(int))
        
        print(f"Epoch {epoch:02d} | Test Acc: {test_accuracy:.4f}")
        scheduler.step(test_accuracy)

    evaluate_and_plot(test_labels.numpy(), test_pred_proba, stage="Final Test")

if __name__ == '__main__':
    main()
