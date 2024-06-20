import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(in_features, hidden_features)
        self.bn1 = BatchNorm(hidden_features)
        self.gc2 = GCNConv(hidden_features, hidden_features)
        self.bn2 = BatchNorm(hidden_features)
        self.gc3 = GCNConv(hidden_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.gc1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gc2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gc3(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)


