# -*- coding: utf-8 -*-
'''
@Time    : 2023/10/15 16:46
@Author  : Zekun Cai
@File    : pred_MemDA_beijing.py
@Software: PyCharm
'''

import os
import shutil
import time
import sys

import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from torch.utils.data import Dataset

from setting.Param import *
from setting.Dataset_Setting import *
from utils.Utils import *
from utils.Metrics import *
from MemDA import *


class TensorDatasetIndex(Dataset):
    def __init__(self, *tensors) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return index, tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def get_key(query, mem_start=0):
    keys = []
    for i in range(1, look_back+1):
        key = query - i*day_interval + mem_start
        keys.append(key)

        key = query - i * day_interval + TIMESTEP_IN + mem_start
        keys.append(key)
    mask = np.where(keys[-2]>=0)
    return keys, mask


def get_value(keys, memory):
    values = []
    for key in keys:
        value = memory[key]
        values.append(value)
    return values


def update_value(idx, new_value, memory, mem_start=0):
    memory[idx + mem_start] = new_value
    return memory


def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * train_ration)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS


def evaluateModel(model, criterion, data_iter, memory, mem_start=0, update=False):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for idx, (x, y) in data_iter:
            keys, _ = get_key(idx, mem_start=mem_start)
            values = get_value(keys, memory)
            x, y = x.to(device), y.to(device)
            values = [value.to(device) for value in values]
            y_pred, new_values = model(x, values)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            memory = update_value(idx, new_values.cpu(), memory, mem_start=mem_start) if update else memory
        return l_sum / n


def predictModel(model, data_iter, memory, mem_start=0, update=False):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in data_iter:
            keys, _ = get_key(idx, mem_start=mem_start)
            values = get_value(keys, memory)
            x, y = x.to(device), y.to(device)
            values = [value.to(device) for value in values]
            YS_pred_batch, new_values = model(x, values)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
            memory = update_value(idx, new_values.cpu(), memory, mem_start=mem_start) if update else memory
        YS_pred = np.vstack(YS_pred)
    return YS_pred


def trainModel(mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS), torch.Tensor(YS)

    # Splitting the training validation set
    trainval_data = TensorDatasetIndex(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, 12, shuffle=False)
    print('TRAIN_len, VAL_len', len(train_data), len(val_data))

    # Replay Memory
    replay_memory = torch.randn((XS.shape[0], encoder_dim, n_node, channel), dtype=torch.float)
    nn.init.xavier_normal_(replay_memory)
    replay_memory[:look_back * day_interval] = torch.zeros((look_back * day_interval, encoder_dim, n_node, channel), dtype=torch.float).to(device)

    # Train
    model = getModel(device, encoder, n_node, channel, TIMESTEP_OUT, adj_path, adj_type)
    summary(model, [(BATCHSIZE, channel, n_node, TIMESTEP_IN)].extend([(BATCHSIZE, encoder_dim, n_node, channel) for _ in range(look_back)]), batch_dim=None, device=device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    min_val_loss = np.inf
    wait = 0

    for epoch in range(EPOCH):
        replay_memory_tmp = replay_memory.clone()
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for idx, (x, y) in train_iter:
            keys, mask = get_key(idx, mem_start=0)
            idx, x, y, keys = idx[mask], x[mask], y[mask], [key[mask] for key in keys]
            values = get_value(keys, replay_memory)  # Recall Backtracking Information
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            values = [value.to(device) for value in values]
            y_pred, new_values = model(x, values)
            replay_memory_tmp = update_value(idx, new_values.cpu(), replay_memory_tmp, mem_start=0)  # Update Replay Memory
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        for i in range(look_back):
            replay_memory_tmp[i*day_interval:(i+1)*day_interval] = replay_memory_tmp[look_back*day_interval:(look_back+1)*day_interval]

        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter, replay_memory_tmp, mem_start=0, update=True)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + MODELNAME + '.pt')
            replay_memory = replay_memory_tmp
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, "validation loss:",
              val_loss)
        with open(PATH + '/' + MODELNAME + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:",
                val_loss))

    # Save
    torch_score = evaluateModel(model, criterion, train_iter, replay_memory, mem_start=0,
                                update=True)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, 12, shuffle=False), replay_memory,
                           mem_start=0, update=True)
    np.save(PATH + '/' + MODELNAME + '_memory.npy', replay_memory[-look_back * day_interval - 1:-1].cpu())
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
    XS_torch, YS_torch = torch.Tensor(XS), torch.Tensor(YS)
    test_data = TensorDatasetIndex(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, 12, shuffle=False)

    # Test
    model = getModel(device, encoder, n_node, channel, TIMESTEP_OUT, adj_path, adj_type)
    model.load_state_dict(torch.load(PATH + '/' + MODELNAME + '.pt'))
    replay_memory_his = torch.Tensor(np.load(PATH + '/' + MODELNAME + '_memory.npy'))
    replay_memory_holder = torch.zeros((XS.shape[0], encoder_dim, n_node, channel), dtype=torch.float)
    replay_memory = torch.cat([replay_memory_his, replay_memory_holder], dim=0)

    torch_score = evaluateModel(model, criterion, test_iter, replay_memory, mem_start=len(replay_memory_his),
                                update=True)
    YS_pred = predictModel(model, test_iter, replay_memory, mem_start=len(replay_memory_his), update=True)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)

    # Save
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10f" % (MODELNAME, mode, torch_score))
    f = open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10f\n" % (MODELNAME, mode, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (
        MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
        MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (
            i + 1, MODELNAME, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
            i + 1, MODELNAME, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())


################# Parameter Setting #######################
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)
# np.random.seed(100)
###########################################################
MODELNAME = 'MemDA'
DATANAME = DATA_SET
n_node = SETTING[DATA_SET]['n_node']
channel = SETTING[DATA_SET]['fea']
flow_path = SETTING[DATA_SET]['data_file']
adj_path = SETTING[DATA_SET]['adj_file']
adj_type = SETTING[DATA_SET]['adj_type']
day_interval = SETTING[DATA_SET]['day_interval']
BATCHSIZE = 8

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
