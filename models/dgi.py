import torch
import torch.nn as nn
from layers import GATNet, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,heads,edge_dim):
        super(DGI, self).__init__()
        self.gat = GATNet(in_channels, hidden_channels,out_channels,heads,edge_dim)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(out_channels)

    def forward(self,data,data_corrupt):
        h_1 = self.gat(data)
        c = self.read(h_1)
        c = self.sigm(c)

        h_2 = self.gat(data_corrupt)
        ret = self.disc(c, h_1, h_2)
        return ret

    # # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = self.gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)

    #     return h_1.detach(), c.detach()

