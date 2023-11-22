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
DATA_SET = 'Electricity'
GPU = '1'
encoder = 'gwn'
look_back = 2
mem_num = 20
mem_dim = 32
ntn_dim = 32
ntn_k = 5
encoder_dim = 256