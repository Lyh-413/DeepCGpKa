import torch

# default device
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_NAME)

# DataSet Parameters
DATA_DIR ='D:\code\pka\ca+model\data'
RESI_TYPE = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# Model Parameters
TOTAL_EPOCH = 400
LR_START = 1e-3
ITER_NUM = 9743
BATCH_SIZE = 1
MANUAL_BATCH_SIZE = 1
NUM_WORKERS = 4
RESTART = 0
SAVE_WEIGHT_DIR = './model_weight'

# Layer Parameters
LAYER_PARA = {
    'n_ResiType': 20,

    '1d_FirstConvChannels': [8,8],
    '1d_FirstConvKernelSize': [15,15],
    '1d_BlockLayers': [3,4,5],
    '1d_BlockChannels': [8,16,32],
    '1d_BlockKernelSize': 15,
    '1d_LastChannel': 5,

    '2d_BlockLayers': [3,4,5],
    '2d_BlockChannels': [16,32,64],
    #'2d_BlockKernelSize': (5,5),
    
    'TriangleMultiplicativeHidden': [16,32,64],
    'TriangleSelfAttentionHead': [2,2,2],
    'TriangleSelfAttentionHeadChannel': [8,16,32],

    'AFDim': [64, 64, 64, 64],
    'AFHeads': [4, 8, 8, 8], #[5, 5, 5]
    'AFHeadDim': [ 8, 16, 16, 8],

    '2d_BlockLayer': [3, 4, 6, 3],
    '2d_BlockChannels': [ 64,128, 256, 128, 64, 32]
}
