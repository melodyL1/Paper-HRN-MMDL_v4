import torch.nn.parallel
from TPPI.loaders.auxil import *
from os.path import join as pjoin
import numpy as np
import pandas as pd
from tool import string_to_int_list

def get_trainLoader_list(cfg, logdir):
    fix_data_path = 'dataset/split_dataset/'

    # 获取训练集和验证集的分批文件列表
    train_files = []
    val_files = []

    # 假设分批文件按命名规则存储，如 x_train_batch_0.npy, y_train_batch_0.npy
    big_data_batch = cfg["Preprocessing"]["big_data_batch"]

    for i in range(big_data_batch):
        x_train_batch = pjoin(fix_data_path, f"x_train_batch{i}.npy")
        y_train_batch = pjoin(fix_data_path, f"y_train_batch{i}.npy")
        train_files.append((x_train_batch, y_train_batch))

        x_val_batch = pjoin(fix_data_path, f"x_val_batch{i}.npy")
        y_val_batch = pjoin(fix_data_path, f"y_val_batch{i}.npy")
        val_files.append((x_val_batch, y_val_batch))

    # 计算类别数和波段数
    numberofclass = cfg['Data']['class']
    bands = sum(string_to_int_list(cfg['Data']['modes_number'])) # (num_patches_in_batch, height, width, num_bands)

    print("number of class is:{}".format(numberofclass))
    print("bands is:{}".format(bands))
    print("load batched dataset list over")

    return train_files, val_files, numberofclass, bands

def get_dataLoader(data, cfg,logdir):
    x_data = np.load(data[0])
    x_data = x_data[:,:,:,cfg["Data"]["band_selection"][0]:cfg["Data"]["band_selection"][1]]
    y_data = np.load(data[1])
    if cfg["Model"] == 'Shallow_Network':
        if cfg["Model_detail"]["Shallow_Multimodal"] == False:
            x_train_reshaped = x_data.reshape((x_data.shape[0], -1))
            concatenated_data = np.concatenate((x_train_reshaped, y_data.reshape(-1, 1)), axis=1)
            x_Shallow_Network = x_train_reshaped
            y_Shallow_Network = y_data.reshape(-1, 1)
            df = pd.DataFrame(concatenated_data, columns=[f'Feature_{i}' for i in range(x_train_reshaped.shape[1])] + ['Label'])
            df.to_csv(logdir + "/data.csv", index=False)
    else:
        x_Shallow_Network = 0
        y_Shallow_Network = 0
    data_hyper = HyperData((np.transpose(x_data, (0, 3, 1, 2)).astype("float32"), y_data)) # 很关键一步，实现格式转换同时实现了numpy格式转换为Tensor类型，这个类型不仅是神经层接受的可求导格式，且可训练优化（因为已经与Variable合并）。

    kwargs = {'num_workers': 1, 'pin_memory': False}
    data_loader = torch.utils.data.DataLoader(data_hyper, batch_size=cfg["Train"]["batch_size"], shuffle=True, **kwargs)
    return data_loader, x_Shallow_Network, y_Shallow_Network