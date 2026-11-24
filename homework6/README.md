# GNN for Classification on PROTEINS (TUDataset)

This project uses a Graph Convolutional Network (GCN) and a Graph Attention Network (GAT) to classify proteins as enzymes or non-enzymes.

## Part 1: Data Loading and Exploration

### Dataset information

|Property|Value|
|---|---|
|Total number of graphs| 1113 |
|Number of classes| 2 |
|Number of node features| 3 |

### First graph (`dataset[0]`)
```
Data(edge_index=[2, 162], x=[42, 3], y=[1])
```
**Node feature matrix `x`** — shape: `(42, 3)`
```
tensor([[1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.]])
```
**Edge index `edge_index`** — shape: `(2, 162)`
```
tensor([[ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,
          4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  8,  9,
          9,  9,  9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13,
         14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18,
         19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22,
         23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28,
         28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32,
         33, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 37, 37,
         37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 41, 41, 41, 41, 41, 41],
        [11, 22, 32, 23, 31, 41, 24, 29, 34,  4, 28, 29, 38, 39,  3, 14, 28, 29,
         38, 39,  6,  7, 26,  5,  7,  8,  5,  6, 19, 25,  6,  9, 10, 25, 35,  8,
         10, 19, 21, 25, 35,  8,  9, 20,  0, 22, 32, 31, 33, 41, 14, 15, 34, 39,
          4, 13, 15, 39, 13, 14, 38, 17, 18, 19, 36, 16, 18, 19, 16, 17, 19, 35,
          7,  9, 16, 17, 18, 20, 21, 35, 10, 19, 21,  9, 19, 20,  0, 11, 30, 40,
          1, 24, 31, 34, 41,  2, 23, 33,  7,  8,  9,  5, 27, 37, 26, 28, 36,  3,
          4, 27,  2,  3,  4, 22, 31, 32,  1, 12, 23, 30, 33, 41,  0, 11, 30, 40,
         12, 24, 31, 34, 41,  2, 13, 23, 33,  8,  9, 18, 19, 16, 27, 37, 26, 36,
         38,  3,  4, 15, 37,  3,  4, 13, 14, 22, 32, 41,  1, 12, 23, 31, 33, 40]])
```
**Graph label `y`**
```
tensor([0])
```
### Average number of nodes per graph

**39.06** nodes per graph (over all 1113 graphs).

## Part 2: Train, Validation, Test Split

|Dataset|Number of graphs|Purpose|
|---|---|---|
|Training|778| Fit model parameters |
|Validation|112| Select each model checkpoint |
|Test|223| Final evaluation only |

- **Random seed:** `random_state=0`

## Part 3: Model Building and Training

### Mini-batches

| Loader | Graphs | Mini-batches |
|---|---|---|
| Training | 778 | 25 |
| Validation | 112 | 4 |
| Test | 223 | 7 |

Example training mini-batch:
```
DataBatch(edge_index=[2, 5608], x=[1545, 3], y=[32], batch=[1545], ptr=[33])
```
- `x` shape: `(1545, 3)` (`[num_nodes, num_features]`)
- `edge_index` shape: `(2, 5608)` (`[2, num_edges]`)
- `batch` shape: `(1545,)` (graph index of each node)
- `y` shape: `(32,)` (one label per graph)

### Model 1: GCN

Two `GCNConv` layers (ReLU, dropout 0.5 after the first), `global_mean_pool`, and a `Linear` output layer.

- `in_channels` = 3, `hidden_channels` = 64, `num_classes` = 2
- Loss: `CrossEntropyLoss`, optimizer: `Adam` (lr = 0.01), epochs: 100, batch size: 32
- **Best checkpoint:** epoch 57 (validation loss = 0.5753)

Training progress (every 10 epochs):
```
Epoch  10 | Train Loss: 0.6131 | Val Loss: 0.6125
Epoch  20 | Train Loss: 0.5946 | Val Loss: 0.6121
Epoch  30 | Train Loss: 0.5870 | Val Loss: 0.6016
Epoch  40 | Train Loss: 0.5898 | Val Loss: 0.6024
Epoch  50 | Train Loss: 0.5904 | Val Loss: 0.6021
Epoch  60 | Train Loss: 0.5826 | Val Loss: 0.5929
Epoch  70 | Train Loss: 0.5777 | Val Loss: 0.5772
Epoch  80 | Train Loss: 0.5776 | Val Loss: 0.5850
Epoch  90 | Train Loss: 0.5781 | Val Loss: 0.5823
Epoch 100 | Train Loss: 0.5843 | Val Loss: 0.5954
```

![GCN loss curves](loss_gcn.png)

### Model 2: GAT

Two `GATConv` layers (first with 4 heads, ELU, dropout 0.5 after the first; second with 1 head and `concat=False`), `global_mean_pool`, and a `Linear` output layer.

- `in_channels` = 3, `hidden_channels` = 64, `heads` = 4, `num_classes` = 2
- Loss: `CrossEntropyLoss`, optimizer: `Adam` (lr = 0.01), epochs: 100, batch size: 32
- **Best checkpoint:** epoch 93 (validation loss = 0.5781)

Training progress (every 10 epochs):
```
Epoch  10 | Train Loss: 0.6424 | Val Loss: 0.6379
Epoch  20 | Train Loss: 0.6050 | Val Loss: 0.6147
Epoch  30 | Train Loss: 0.5916 | Val Loss: 0.6099
Epoch  40 | Train Loss: 0.6041 | Val Loss: 0.5944
Epoch  50 | Train Loss: 0.5971 | Val Loss: 0.6967
Epoch  60 | Train Loss: 0.5876 | Val Loss: 0.6573
Epoch  70 | Train Loss: 0.5849 | Val Loss: 0.6019
Epoch  80 | Train Loss: 0.5965 | Val Loss: 0.6322
Epoch  90 | Train Loss: 0.5962 | Val Loss: 0.5856
Epoch 100 | Train Loss: 0.5897 | Val Loss: 0.5836
```

![GAT loss curves](loss_gat.png)
