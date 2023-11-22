# General
TIMESTEP_IN = 12
TIMESTEP_OUT = 12
BATCHSIZE = 64
LEARN = 0.001
EPOCH = 200
PATIENCE = 10
OPTIMIZER = 'Adam'
LOSS = 'MAE'
TRAINVALSPLIT = 0.2


# MemDA
DATA_SET = 'PeMS20'
GPU = '0'
encoder = 'gwn'
encoder_dim = 256
look_back = 7
mem_num = 20
mem_dim = 32
ntn_dim = 32
ntn_k = 5