import os
import scipy.io as sio
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv
import matplotlib.pyplot as plt
import torch
import yaml
import argparse
from collections import defaultdict

def get_device():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    if cfg['GPU'] is not None:
        if torch.cuda.is_available():
            device = torch.device("cuda", cfg["GPU"])
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist

def img_accuracy(output, target):
    target = target.flatten()
    output = output.flatten()
    hist = _fast_hist(label_true=target, label_pred = output, n_class=16)
    acc = np.diag(hist).sum() / hist.sum()
    return acc

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_pascal_labels():
    return np.asarray(
        [
            [255, 255, 255],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )

def decode_segmap(pred, plot=True):
    label_colours = get_pascal_labels()
    r = pred.copy()
    g = pred.copy()
    b = pred.copy()
    for ll in range(0, 18):
        r[pred == ll] = label_colours[ll, 0]
        g[pred == ll] = label_colours[ll, 1]
        b[pred == ll] = label_colours[ll, 2]
    rgb = np.zeros((pred.shape[0], pred.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.show(block=True)
    else:
        return rgb

# Padding补充
def padWithZeros(X, margin):
    """
    :param X: input, shape:[H,W,C]
    :param margin: padding
    :return: new data, shape:[H+2*margin, W+2*margin, C]
    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
# 生成PP
def create_PP(cfg, X, y):
    # 清空缓存数据
    if cfg['Root_path'] == 0:
        root_path = ''
    else:
        root_path = cfg['Root_path']
    cache_dir = root_path + 'dataset/data_cache_pool'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print("Cache directory has been cleared and is ready for use.")
    # 参数加载
    windowSize = cfg['Preprocessing']['PP_size']
    big_data_batch = cfg['Preprocessing']["big_data_batch"]
    # big_data_batch = 5

    margin = (windowSize - 1) // 2
    zeroPaddedX = padWithZeros(X, margin = margin)

    # 获取每个 batch 中像素点的坐标（均衡类别分布）
    h, w = y.shape[:2]
    all_coords = [(r, c) for r in range(h) for c in range(w)]
    batch_indices_list = []
    batch_size = len(all_coords) // big_data_batch
    for i in range(big_data_batch):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < big_data_batch - 1 else len(all_coords)
        batch_indices_list.append(all_coords[start_idx:end_idx])

    file_names = []
    all_patches_locations = []

    for batch_idx, coords in enumerate(batch_indices_list):
        patchesData = np.zeros((len(coords), windowSize, windowSize, X.shape[2]))
        patchesLocations_batch = []

        for i, (r, c) in enumerate(coords):
            patch = zeroPaddedX[r : r + 2*margin + 1, c : c + 2*margin + 1]
            patchesData[i] = patch
            patchesLocations_batch.append([r, c, f"batch_{batch_idx}.mat"])

        # 保存为 .mat 文件
        file_name = os.path.join(cache_dir, f"batch_{batch_idx}.mat")
        sio.savemat(file_name, {'patchesData': patchesData})

        file_names.append(file_name)
        all_patches_locations.extend(patchesLocations_batch)

    all_patches_locations = np.array(all_patches_locations)
    return file_names, all_patches_locations
# 旧的写法
# def padWithZeros(X, margin):
#     if margin == 0:
#         return X
#     else:
#         newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
#         x_offset = margin
#         y_offset = margin
#         newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
#         return newX
#
# def creat_PP(windowSize, X):
#     margin = int((windowSize - 1) / 2)
#     zeroPaddedX = padWithZeros(X, margin=margin)
#     patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
#     patchIndex = 0
#     for r in range(margin, zeroPaddedX.shape[0] - margin):
#         for c in range(margin, zeroPaddedX.shape[1] - margin):
#             patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
#             patchesData[patchIndex, :, :, :] = patch
#             patchIndex = patchIndex + 1
#     return patchesData