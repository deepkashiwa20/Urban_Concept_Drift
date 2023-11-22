# -*- coding: utf-8 -*-
'''
@Time    : 2023/10/15 16:46
@Author  : Zekun Cai
@File    : pred_MemDA.py
@Software: PyCharm
'''

import os
import shutil
import time
import sys

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from torch.utils.data import Dataset

from encoder.Encoder_GWN import *
from setting.Param import *
from setting.Dataset_Setting import *
from utils.Utils import *
from utils.Metrics import *


def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * train_ration)
    XS, YS = [], []
    XS_C, YS_C = [], []
    XS_C2, YS_C2 = [], []
    if mode == 'TRAIN':
        for i in range(look_back * day_interval, TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT]
            XS.append(x), YS.append(y)

            x_c = data[i - day_interval:i - day_interval + TIMESTEP_IN]
            y_c = data[i - day_interval + TIMESTEP_IN:i - day_interval + TIMESTEP_IN + TIMESTEP_OUT]
            XS_C.append(x_c), YS_C.append(y_c)

            x_c2 = data[i - day_interval * 2:i - day_interval * 2 + TIMESTEP_IN]
            y_c2 = data[i - day_interval * 2 + TIMESTEP_IN:i - day_interval * 2 + TIMESTEP_IN + TIMESTEP_OUT]
            XS_C2.append(x_c2), YS_C2.append(y_c2)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)

            x_c = data[i - day_interval:i - day_interval + TIMESTEP_IN]
            y_c = data[i - day_interval + TIMESTEP_IN:i - day_interval + TIMESTEP_IN + TIMESTEP_OUT]
            XS_C.append(x_c), YS_C.append(y_c)

            x_c2 = data[i - day_interval * 2:i - day_interval * 2 + TIMESTEP_IN]
            y_c2 = data[i - day_interval * 2 + TIMESTEP_IN:i - day_interval * 2 + TIMESTEP_IN + TIMESTEP_OUT]
            XS_C2.append(x_c2), YS_C2.append(y_c2)
    XS, YS, XS_C, YS_C, XS_C2, YS_C2 = np.array(XS), np.array(YS), np.array(XS_C), np.array(YS_C), np.array(XS_C2), np.array(YS_C2)
    XS, YS, XS_C, YS_C, XS_C2, YS_C2 = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis], \
                                       XS_C[:, :, :, np.newaxis], YS_C[:, :, :, np.newaxis], \
                                       XS_C2[:, :, :, np.newaxis], YS_C2[:, :, :, np.newaxis]
    XS = np.concatenate((XS, XS_C, YS_C, XS_C2, YS_C2), axis=-1)
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS


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
        self.end_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class memda_net_plain(nn.Module):
    def __init__(self, device, num_nodes=320, in_dim=1, supports=None):
        super(memda_net_plain, self).__init__()
        self.encoder = gwnet_head(device, num_nodes=num_nodes, in_dim=in_dim, supports=supports)

        self.memory = nn.Parameter(torch.randn(20, 32), requires_grad=True)
        nn.init.xavier_normal_(self.memory)
        self.mem_proj1 = nn.Linear(in_features=256, out_features=32)
        self.mem_proj2 = nn.Linear(in_features=32, out_features=256)

        self.cts_proj1 = nn.Linear(in_features=256, out_features=32)
        self.cts_proj2 = nn.Linear(in_features=25, out_features=256)

        self.NTN1 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN2 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN3 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN4 = NTN(input_dim=32, feature_map_dim=5)
        self.NTN5 = NTN(input_dim=64, feature_map_dim=5)

        self.meta_W = nn.Linear(256, 10)
        self.meta_b = nn.Linear(256, 1)

        self.decoder = Decoder(in_dim=256, hidden_dim=512, out_dim=12)
        self.merge = nn.Linear(in_features=10, out_features=1)

    def forward(self, input):
        x = input[:, 0:1, :, :]
        x_c = input[:, 1:2, :, :]
        y_c = input[:, 2:3, :, :]
        x_c2 = input[:, 3:4, :, :]
        y_c2 = input[:, 4:5, :, :]

        hidden = self.encoder(x)
        hidden_xc = self.encoder(x_c)
        hidden_yc = self.encoder(y_c)
        hidden_xc2 = self.encoder(x_c2)
        hidden_yc2 = self.encoder(y_c2)

        hidden_merge = torch.cat((hidden, hidden_xc, hidden_yc, hidden_xc2, hidden_yc2), dim=-1)

        hidden_merge_re = hidden_merge.permute(0, 2, 3, 1)
        query = self.mem_proj1(hidden_merge_re)
        att = torch.softmax(torch.matmul(query, self.memory.t()), dim=-1)
        res_mem = torch.matmul(att, self.memory)
        res_mem = self.mem_proj2(res_mem)
        res_mem = res_mem.permute(0, 3, 1, 2)

        hidden_cts = self.cts_proj1(hidden_merge_re)
        px_1 = self.NTN1(hidden_cts[:, :, 0, :], hidden_cts[:, :, 1, :])
        px_2 = self.NTN2(hidden_cts[:, :, 1, :], hidden_cts[:, :, 3, :])
        px_3 = self.NTN3(hidden_cts[:, :, 0, :], hidden_cts[:, :, 3, :])
        py_1 = self.NTN4(hidden_cts[:, :, 2, :], hidden_cts[:, :, 4, :])
        xy_1 = hidden_cts[:, :, 1:3, :].reshape(-1, n_node, 2 * 32)
        xy_2 = hidden_cts[:, :, 3:5, :].reshape(-1, n_node, 2 * 32)
        pxy_1 = self.NTN5(xy_1, xy_2)

        sim_mtx = torch.cat([px_1, px_2, px_3, py_1, pxy_1], dim=-1)
        sim_mtx = self.cts_proj2(sim_mtx)

        W = self.meta_W(sim_mtx)
        b = self.meta_b(sim_mtx)
        W = torch.reshape(W, (-1, n_node, 10, 1))
        b = b.view(-1, 1, n_node, 1)
        W = torch.softmax(W, dim=-2)
        hidden = torch.cat((hidden_merge, res_mem), dim=-1)
        hidden = torch.einsum("bdnt,bntj->bdnj", [hidden, W]) + b

        output = self.decoder(hidden)
        return output


def getModel():
    if adj_path:
        adj_mx = load_adj(adj_path, adj_type)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
    else:
        supports = None
    model = memda_net_plain(device, num_nodes=n_node, in_dim=channel, supports=supports).to(device)
    return model


def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred


def trainModel(mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)

    # Splitting the training validation set
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=False)
    print('TRAIN_len, VAL_len', len(train_data), len(val_data))

    # Train
    model = getModel()
    summary(model, ((BATCHSIZE, channel * 5, n_node, TIMESTEP_IN)), batch_dim=None, device=device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    min_val_loss = np.inf
    wait = 0

    for epoch in range(EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + MODELNAME + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + MODELNAME + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))

    # Save
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10f" % (MODELNAME, mode, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
    with open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10f\n" % (MODELNAME, mode, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (MODELNAME, mode, MSE, RMSE, MAE, MAPE))


def testModel(mode, XS, YS):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    criterion = nn.L1Loss()
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel()
    model.load_state_dict(torch.load(PATH + '/' + MODELNAME + '.pt'))

    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)

    # Save
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10f" % (MODELNAME, mode, torch_score))
    f = open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10f\n" % (MODELNAME, mode, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i + 1, MODELNAME, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i + 1, MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())


################# Parameter Setting #######################
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
###########################################################
MODELNAME = 'MemDA_Plain'
DATANAME = DATA_SET
n_node = SETTING[DATA_SET]['n_node']
channel = SETTING[DATA_SET]['fea']
flow_path = SETTING[DATA_SET]['data_file']
adj_path = SETTING[DATA_SET]['adj_file']
adj_type = SETTING[DATA_SET]['adj_type']
day_interval = SETTING[DATA_SET]['day_interval']

KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + 'c{}_mn{}_md{}_nd{}_nk{}_'.format(look_back, mem_num, mem_dim, ntn_dim, ntn_k) + datetime.now().strftime("%y%m%d%H%M")
PATH = './save/save_{}/'.format(DATANAME) + KEYWORD
train_ration = get_train_ratio(SETTING[DATA_SET])
data = pd.read_csv(flow_path, index_col=0).values
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)


###########################################################


def main():
    currentPython = sys.argv[0]
    shutil.copytree('encoder', PATH + '/encoder')
    shutil.copytree('setting', PATH + '/setting', dirs_exist_ok=True)
    shutil.copytree('utils', PATH + '/utils', dirs_exist_ok=True)
    shutil.copy2(currentPython, PATH)
    shutil.copy2('MemDA.py', PATH)

    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS.shape', trainXS.shape, trainYS.shape)
    trainModel('train', trainXS, trainYS)

    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel('test', testXS, testYS)


if __name__ == '__main__':
    main()
