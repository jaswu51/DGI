import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T


import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
import copy
# importing pandas
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.abspath('/Users/xiaokeai/Documents/GitHub/DGI_fork/'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, GATConv
from torch_geometric.utils import degree
import random
from utils import KarateDataset,KarateDatasetCorrupt

# GAT model
class GATNet(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels,heads,edge_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,heads,edge_dim)
        self.conv2= GATConv(hidden_channels*heads, hidden_channels,heads,edge_dim)
        self.lineardropout = nn.Sequential(
            nn.Linear(hidden_channels*heads, hidden_channels), 
            nn.Dropout(0.1), 
            nn.Linear(hidden_channels, out_channels))

    def forward(self, data):
        x=self.conv1(data.x,data.edge_index,data.edge_attr)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        x=self.conv2(x,data.edge_index,data.edge_attr)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        x=self.lineardropout(x)
        return F.log_softmax(x, dim=1)
