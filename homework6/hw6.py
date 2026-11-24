import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split

readme = []   # used to append outputs from every section

def add(section):
    readme.append(section.strip() + "\n")

### Part 1: Data loading and exploration
dataset = TUDataset(root="data/TUDataset", name="PROTEINS")

# Dataset information
num_graphs = len(dataset)
num_classes = dataset.num_classes
num_node_features = dataset.num_node_features
g = dataset[0]
avg_nodes = sum(d.num_nodes for d in dataset) / num_graphs

add(f"""# GNN for Classification on PROTEINS (TUDataset)

This project uses a Graph Convolutional Network (GCN) and a Graph Attention Network (GAT) to classify proteins as enzymes or non-enzymes.""")

add(f"""## Part 1: Data Loading and Exploration

### Dataset information

|Property|Value|
|---|---|
|Total number of graphs| {num_graphs} |
|Number of classes| {num_classes} |
|Number of node features| {num_node_features} |

### First graph (`dataset[0]`)
```
{g}
```
**Node feature matrix `x`** — shape: `{tuple(g.x.shape)}`
```
{g.x}
```
**Edge index `edge_index`** — shape: `{tuple(g.edge_index.shape)}`
```
{g.edge_index}
```
**Graph label `y`**
```
{g.y}
```
### Average number of nodes per graph

**{avg_nodes:.2f}** nodes per graph (over all {num_graphs} graphs).""")

print("Data exploration finished")

### Part 2: Data preprocessing
SEED = 0
indices = list(range(num_graphs))

trainval_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED)
train_idx, val_idx = train_test_split(trainval_idx, test_size=0.125, random_state=SEED)

assert not (set(train_idx) & set(val_idx))
assert not (set(train_idx) & set(test_idx))
assert not (set(val_idx) & set(test_idx))

train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]

add(f"""## Part 2: Train, Validation, Test Split

|Dataset|Number of graphs|Purpose|
|---|---|---|
|Training|{len(train_dataset)}| Fit model parameters |
|Validation|{len(val_dataset)}| Select each model checkpoint |
|Test|{len(test_dataset)}| Final evaluation only |

- **Random seed:** `random_state={SEED}`""")

print("Data preprocessing finished")

### Part 3: Model Building and Training

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mini-batches
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

batch0 = next(iter(train_loader))

add(f"""## Part 3: Model Building and Training

### Mini-batches

| Loader | Graphs | Mini-batches |
|---|---|---|
| Training | {len(train_dataset)} | {len(train_loader)} |
| Validation | {len(val_dataset)} | {len(val_loader)} |
| Test | {len(test_dataset)} | {len(test_loader)} |

Example training mini-batch:
```
{batch0}
```
- `x` shape: `{tuple(batch0.x.shape)}` (`[num_nodes, num_features]`)
- `edge_index` shape: `{tuple(batch0.edge_index.shape)}` (`[2, num_edges]`)
- `batch` shape: `{tuple(batch0.batch.shape)}` (graph index of each node)
- `y` shape: `{tuple(batch0.y.shape)}` (one label per graph)""")


# ---------------- Model definitions ----------------
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ---------------- Training helpers ----------------
def evaluate_loss(model, loader, criterion):
    """Average loss per graph, no gradients, no parameter updates."""
    model.eval()
    total_loss, total_graphs = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs
    return total_loss / total_graphs


def train_model(model, name, epochs=100, lr=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, log_lines = [], [], []
    best_val, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_graphs = 0.0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs

        train_loss = total_loss / total_graphs
        val_loss = evaluate_loss(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # keep the checkpoint with the lowest validation loss
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            line = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            print(f"[{name}] {line}")
            log_lines.append(line)

    torch.save(best_state, f"best_{name.lower()}.pt")
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "log": "\n".join(log_lines),
    }


def plot_losses(res, name):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(res["train_losses"]) + 1), res["train_losses"], label="Training loss")
    plt.plot(range(1, len(res["val_losses"]) + 1), res["val_losses"], label="Validation loss")
    plt.axvline(res["best_epoch"], color="gray", linestyle="--", label=f"Best epoch ({res['best_epoch']})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name}: training and validation loss")
    plt.legend()
    plt.tight_layout()
    fname = f"loss_{name.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


HIDDEN = 64
EPOCHS = 100
LR = 0.01

# ---------------- Model 1: GCN ----------------
gcn = GCNClassifier(dataset.num_features, HIDDEN, dataset.num_classes)
gcn_res = train_model(gcn, "GCN", epochs=EPOCHS, lr=LR)
gcn_plot = plot_losses(gcn_res, "GCN")

add(f"""### Model 1: GCN

Two `GCNConv` layers (ReLU, dropout 0.5 after the first), `global_mean_pool`, and a `Linear` output layer.

- `in_channels` = {dataset.num_features}, `hidden_channels` = {HIDDEN}, `num_classes` = {dataset.num_classes}
- Loss: `CrossEntropyLoss`, optimizer: `Adam` (lr = {LR}), epochs: {EPOCHS}, batch size: {BATCH_SIZE}
- **Best checkpoint:** epoch {gcn_res['best_epoch']} (validation loss = {gcn_res['best_val']:.4f})

Training progress (every 10 epochs):
```
{gcn_res['log']}
```

![GCN loss curves]({gcn_plot})""")

# ---------------- Model 2: GAT ----------------
HEADS = 4
gat = GATClassifier(dataset.num_features, HIDDEN, dataset.num_classes, heads=HEADS)
gat_res = train_model(gat, "GAT", epochs=EPOCHS, lr=LR)
gat_plot = plot_losses(gat_res, "GAT")

add(f"""### Model 2: GAT

Two `GATConv` layers (first with {HEADS} heads, ELU, dropout 0.5 after the first; second with 1 head and `concat=False`), `global_mean_pool`, and a `Linear` output layer.

- `in_channels` = {dataset.num_features}, `hidden_channels` = {HIDDEN}, `heads` = {HEADS}, `num_classes` = {dataset.num_classes}
- Loss: `CrossEntropyLoss`, optimizer: `Adam` (lr = {LR}), epochs: {EPOCHS}, batch size: {BATCH_SIZE}
- **Best checkpoint:** epoch {gat_res['best_epoch']} (validation loss = {gat_res['best_val']:.4f})

Training progress (every 10 epochs):
```
{gat_res['log']}
```

![GAT loss curves]({gat_plot})""")

#Write README.md
with open("README.md", "w") as f:
    f.write("\n".join(readme))

print("README.md written")