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



import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, GATConv
from torch_geometric.utils import degree
import random

# read text file into pandas DataFrame
df = pd.read_csv("data/Ave.txt", sep=",",header=0)

# preprocessing the df
namelist=  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear']
indexlist=[*range(0, 78, 1)]
mapdict = {indexlist[i]: namelist[i] for i in range(len(indexlist))}

df['path'] = df['path'].str.extract(r'(\d+)(?!.*\d)')
df['path'].astype(int)
df['video_no'].astype(int)
df['x1'].astype(float)
df['x2'].astype(float)
df['y1'].astype(float)
df['y2'].astype(float)
df['class_name']=df['detclass']
df['class_name']=df.class_name.map(mapdict)
df['centroid']= list(zip((df['x1'] + df['x2'])*0.5, (df['y1'] + df['y2'])*0.5))
node_features=np.array(df.drop(['centroid', 'path','class_name','node_id','video_amount','frame_amount','video_no','frame_no','conf'], axis=1)).astype(float)



my_list = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
# random.choices([0,1], k=40)
node_labels=torch.from_numpy(np.array(my_list))

edge_index_from=[]
edge_index_to=[]
edge_attr=[]

# print(df.columns.get_loc('node_id'))
for i in range(node_features.shape[0]):
  for j in range(node_features.shape[0]):
    if i!=j:
      if df.iloc[j,3]-df.iloc[i,3] in [-1,1] and df.iloc[j,5]-df.iloc[i,5]==0:
        edge_index_from.append(i)
        edge_index_to.append(j)
        edge_attr.append([1,distance.euclidean(df.iloc[i,-1],df.iloc[j,-1])])
      elif df.iloc[j,3]-df.iloc[i,3]==0:
        edge_index_from.append(i)
        edge_index_to.append(j)
        edge_attr.append([0,distance.euclidean(df.iloc[i,-1],df.iloc[j,-1])])




# custom dataset
class KarateDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(KarateDataset, self).__init__('.', transform, None, None)
        data = Data()
        data.x = torch.from_numpy(node_features).to(torch.float32)
        data.y=node_labels
        data.edge_attr= torch.tensor(edge_attr)
        data.edge_index= torch.tensor([edge_index_from,edge_index_to])

        train_mask = torch.rand(data.num_nodes) < 0.8
        test_mask = ~train_mask
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        data.num_classes = 2
        data.edge_dim=data.edge_attr.shape[1]
        data.num_classes=2
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class KarateDatasetCorrupt(InMemoryDataset):
    def __init__(self, transform=None):
        super(KarateDatasetCorrupt, self).__init__('.', transform, None, None)
        data = Data()
        idx = np.random.permutation(node_features.shape[0])
        shuf_fts = node_features[idx, :]
        data.x = torch.from_numpy(shuf_fts).to(torch.float32)
        data.y=node_labels
        data.edge_attr= torch.tensor(edge_attr)
        data.edge_index= torch.tensor([edge_index_from,edge_index_to])

        train_mask = torch.rand(data.num_nodes) < 0.8
        test_mask = ~train_mask
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        data.num_classes = 2
        data.edge_dim=data.edge_attr.shape[1]
        data.num_classes=2
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
