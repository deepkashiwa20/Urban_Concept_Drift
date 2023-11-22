# -*- coding: utf-8 -*-
'''
@Time    : 2023/10/15 18:43
@Author  : Zekun Cai
@File    : MemDA.py
@Software: PyCharm
'''
import torch
import torch.nn as nn
from setting.Param import *
from encoder.Encoder_GWN import *
from utils.Utils import *


class NTN(nn.Module):
    def __init__(self, input_dim, feature_map_dim):
        super(NTN, self).__init__()
        self.interaction_dim = feature_map_dim
        self.V = nn.Parameter(torch.randn(feature_map_dim, input_dim * 2, 1), requires_grad=True)
        nn.init.xavier_normal_(self.V)
        self.W1 = nn.Parameter(torch.randn(feature_map_dim, input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W1)
        self.W2 = nn.Parameter(torch.randn(feature_map_dim, input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W2)
        self.b = nn.Parameter(torch.zeros(feature_map_dim), requires_grad=True)

    def forward(self, x_1, x_2):
        feature_map = []
        for i in range(self.interaction_dim):
            x_1_t = torch.matmul(x_1, self.W1[i])
            x_2_t = torch.matmul(x_2, self.W2[i])
            part1 = torch.cosine_similarity(x_1_t, x_2_t, dim=-1).unsqueeze(dim=-1)
            part2 = torch.matmul(torch.cat([x_1, x_2], dim=-1), self.V[i])
            fea = part1 + part2 + self.b[i]
            feature_map.append(fea)
        feature_map = torch.cat(feature_map, dim=-1)
        return torch.relu(feature_map)


class Decoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()
        self.end_conv_1 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=hidden_dim,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class Embed_Trans(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Embed_Trans, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_dim,
                                out_channels=hidden_dim,
                                kernel_size=(1, 1),
                                bias=True)

        self.conv_2 = nn.Conv2d(in_channels=hidden_dim,
                                out_channels=out_dim,
                                kernel_size=(1, 1),
                                bias=True)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        return x


class memda_net(nn.Module):
    def __init__(self, device, encoder='gwn', num_nodes=0, channel=0, out=0, supports=None):
        super(memda_net, self).__init__()
        self.num_nodes = num_nodes
        if encoder == 'gwn':
            self.encoder = gwnet_head(device, num_nodes=num_nodes, in_dim=channel, supports=supports,
                                      skip_channels=encoder_dim)
        else:
            raise NameError('Encoder Undefined')

        self.emd_trans = nn.ModuleList([Embed_Trans(in_dim=encoder_dim, hidden_dim=64, out_dim=encoder_dim) for _ in range(2*look_back)])

        # Memory Modules
        self.pattern_memory = nn.Parameter(torch.randn(mem_num, mem_dim), requires_grad=True)
        nn.init.xavier_normal_(self.pattern_memory)
        self.mem_proj1 = nn.Linear(in_features=encoder_dim, out_features=mem_dim)
        self.mem_proj2 = nn.Linear(in_features=mem_dim, out_features=encoder_dim)

        # Drift Adaptation Modules
        self.look_back = look_back
        self.ntn_dim = ntn_dim
        self.ntn_dim = ntn_k
        self.drift_x1 = nn.ModuleList([NTN(input_dim=ntn_dim, feature_map_dim=ntn_k) for _ in range(look_back)])
        self.drift_x2 = nn.ModuleList([NTN(input_dim=ntn_dim, feature_map_dim=ntn_k) for _ in range(look_back - 1)])
        self.drift_y = nn.ModuleList([NTN(input_dim=ntn_dim, feature_map_dim=ntn_k) for _ in range(look_back - 1)])
        self.drift_xy = nn.ModuleList([NTN(input_dim=2 * ntn_dim, feature_map_dim=ntn_k) for _ in range(look_back - 1)])

        self.drift_num = 4 * look_back - 3
        self.drift_dim = encoder_dim
        self.drift_proj1 = nn.Linear(in_features=encoder_dim, out_features=ntn_dim)
        self.drift_proj2 = nn.Linear(in_features=self.drift_num * ntn_k, out_features=self.drift_dim)

        self.meta_num = (2 * look_back + 1) * 2
        self.meta_W = nn.Linear(self.drift_dim, self.meta_num)
        self.meta_b = nn.Linear(self.drift_dim, 1)

        self.decoder = Decoder(in_dim=encoder_dim, hidden_dim=512, out_dim=out)

    def forward(self, x, embedding):
        hidden_x = self.encoder(x)
        for idx, layer in enumerate(self.emd_trans):
            embedding[idx] = layer(embedding[idx])
        hidden_merge = torch.cat([hidden_x] + embedding, dim=-1)

        hidden_merge_re = hidden_merge.permute(0, 2, 3, 1)
        query = self.mem_proj1(hidden_merge_re)
        att = torch.softmax(torch.matmul(query, self.pattern_memory.t()), dim=-1)
        res_mem = torch.matmul(att, self.pattern_memory)
        res_mem = self.mem_proj2(res_mem)
        res_mem = res_mem.permute(0, 3, 1, 2)

        # hidden_cts [x,x,y,x,y,x,y,x,y,...]
        # index      [0,1,2,3,4,5,6,7,8,...]
        hidden_cts = self.drift_proj1(hidden_merge_re)
        drift_results = []
        for i in range(self.look_back):
            if 2*i+1<self.meta_num//2:
                drift_results.append(
                    self.drift_x1[i](hidden_cts[:, :, 0, :], hidden_cts[:, :, 2 * i + 1, :]))  # ([0,1],[0,3]...)
            if 2 * i + 4 < self.meta_num // 2:
                drift_results.append(
                    self.drift_x2[i](hidden_cts[:, :, 2 * i + 1, :], hidden_cts[:, :, 2 * i + 3, :]))  # ([1,3],[3,5]...)
                drift_results.append(
                    self.drift_y[i](hidden_cts[:, :, 2 * i + 2, :], hidden_cts[:, :, 2 * i + 4, :]))  # ([2,4],[4,6]...)
                drift_results.append(self.drift_xy[i]  # ([12,34],[12,56])
                                     (hidden_cts[:, :, 1:3, :].reshape(-1, self.num_nodes, 2 * ntn_dim),
                                      hidden_cts[:, :, 2 * i + 3:2 * i + 5, :].reshape(-1, self.num_nodes, 2 * ntn_dim)))

        drift_mtx = torch.cat(drift_results, dim=-1)
        drift_mtx = self.drift_proj2(drift_mtx)

        W = self.meta_W(drift_mtx)
        b = self.meta_b(drift_mtx)
        W = torch.reshape(W, (-1, self.num_nodes, self.meta_num, 1))
        b = b.view(-1, 1, self.num_nodes, 1)
        W = torch.softmax(W, dim=-2)
        hidden = torch.cat((hidden_merge, res_mem), dim=-1)
        hidden = torch.einsum("bdnt,bntj->bdnj", [hidden, W]) + b

        output = self.decoder(hidden)
        return output, hidden_x.detach()


def getModel(device, encoder='gwn', n_node=0, channel=0, out=0, adj_path=None, adj_type=None):
    if adj_path:
        adj_mx = load_adj(adj_path, adj_type)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
    else:
        supports = None
    model = memda_net(device, encoder=encoder, num_nodes=n_node, channel=channel, out=out, supports=supports).to(device)
    return model
