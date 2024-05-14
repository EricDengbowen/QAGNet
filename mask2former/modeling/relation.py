import torch
from torch import nn
from detectron2.layers import Linear
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features

        self.out_features = out_features

        self.dropout=nn.Dropout(p=dropout)
        self.alpha = alpha
        self.concat = concat

        self.linear1=Linear(self.in_features,self.out_features)
        weight_init.c2_xavier_fill(self.linear1)
        self.linear2 = Linear(2 * self.out_features, 1)
        weight_init.c2_xavier_fill(self.linear2)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = self.linear1(inp)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(self.linear2(a_input).squeeze(2))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]

        attention = F.softmax(attention, dim=1)

        attention=self.dropout(attention)

        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
