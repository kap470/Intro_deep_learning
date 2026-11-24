### Part 1: Data loading and exploration
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root="data/TUDataset", name="PROTEINS")

# Dataset information
num_graphs = len(dataset)
num_classes = dataset.num_classes
num_node_features = dataset.num_node_features

# First graph
g = dataset[0]

# --- Average nodes per graph ---
avg_nodes = sum(d.num_nodes for d in dataset) / num_graphs

md = f"""# GNN for Classification on PROTEINS

```
In this task it is used Graph Convolutional Network (GCN) and a Graph Attention Network (GAT) for protein classification
```

## Part 1: Data Loading and Exploration

### Dataset information

| Property | Value |
|---|---|
| Dataset | `PROTEINS` (TUDataset) |
| Total number of graphs | {num_graphs} |
| Number of classes | {num_classes} |
| Number of node features | {num_node_features} |

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

**{avg_nodes:.2f}** nodes per graph (over all {num_graphs} graphs).
"""

with open("README.md", "w") as f:
    f.write(md)

print(md)