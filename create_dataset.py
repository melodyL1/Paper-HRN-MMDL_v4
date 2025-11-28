import os
from os.path import join as pjoin
import shutil
import scipy.io as sio
import yaml
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from osgeo import gdal
from collections import defaultdict

# 工具箱函数
# 字符串转整型
def string_to_int_list(string):
    # 使用逗号分隔字符串
    str_list = string.split(',')
    # 初始化整数列表
    int_list = []

    # 遍历字符串列表
    for item in str_list:
        # 如果字符串表示true，则转换为1
        if item.lower() == "true":
            int_list.append(1)
        # 如果字符串表示false，则转换为0
        elif item.lower() == "false":
            int_list.append(0)
        # 否则，假设字符串是数字，直接转换为整数
        else:
            int_list.append(int(item))
    return int_list
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
# 批数据均衡化
def balanced_batch_indices(y, big_data_batch, num_classes):
    """
    将每个类别的像素均匀分配到多个 batch 中
    """
    class_indices = defaultdict(list)
    h, w = y.shape[:2]

    # 收集每类标签对应的像素位置
    for r in range(h):
        for c in range(w):
            label = int(y[r, c])  # 确保为整数
            if 0 < label <= num_classes:
                class_indices[label].append((r, c))

    # 初始化每个 batch 的像素索引列表
    batch_indices_list = [[] for _ in range(big_data_batch)]

    # 将每类标签的位置打乱，并均匀分配到每个 batch
    for cls in class_indices:
        positions = class_indices[cls]
        np.random.shuffle(positions)
        for idx, pos in enumerate(positions):
            batch_id = idx % big_data_batch
            batch_indices_list[batch_id].append(pos)

    return batch_indices_list

# 核心处理函数
# 归一化
def normalizations(data):
    for i in range(data.shape[2]):
        min_value = np.min(data[:,:,i][data[:,:,i] > 0])
        max_value = np.max(data[:,:,i][data[:,:,i] > 0])
        data[:,:,i][data[:,:,i] > 0] = (data[:,:,i][data[:,:,i] > 0] - min_value) / (max_value - min_value)
    return data
# 生成PP
def create_PP(cfg, X, y):
    windowSize = cfg['Preprocessing']['PP_size']
    removeZeroLabels = cfg['Preprocessing']["y_remove_zeros"]
    big_data_batch = cfg['Preprocessing']["big_data_batch"]
    num_classes = cfg['Data']["class"]

    margin = (windowSize - 1) // 2
    zeroPaddedX = padWithZeros(X, margin=margin)

    # 获取每个 batch 中像素点的坐标（均衡类别分布）
    batch_indices_list = balanced_batch_indices(y, big_data_batch, num_classes)

    file_names = []
    all_patches_locations = []

    for batch_idx, coords in enumerate(batch_indices_list):
        patchesData = np.zeros((len(coords), windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((len(coords)))
        patchesLocations_batch = []

        for i, (r, c) in enumerate(coords):
            patch = zeroPaddedX[r : r + 2*margin + 1, c : c + 2*margin + 1]
            patchesData[i] = patch
            patchesLabels[i] = y[r, c]
            patchesLocations_batch.append([r, c, f"batch_{batch_idx}.mat"])

        # 处理标签值（去除0并从1开始变为0）
        if removeZeroLabels:
            valid_indices = (patchesLabels > 0) & (patchesLabels <= num_classes)
            patchesData = patchesData[valid_indices]
            patchesLabels = patchesLabels[valid_indices]
            patchesLocations_batch = np.array(patchesLocations_batch)[valid_indices]
            patchesLabels -= 1

        # 保存为 .mat 文件
        os.makedirs(cache_dir, exist_ok=True)
        file_name = os.path.join(cache_dir, f"batch_{batch_idx}.mat")
        sio.savemat(file_name, {'patchesData': patchesData, 'patchesLabels': patchesLabels})

        file_names.append(file_name)
        all_patches_locations.extend(patchesLocations_batch)

    all_patches_locations = np.array(all_patches_locations)
    return file_names, all_patches_locations
# 数据集分割
def split_data(pixels, labels, indexes, percent, rand_state):
    pixels_number = np.unique(labels, return_counts=1)[1]
    train_set_size = [int(np.ceil(a * percent)) for a in pixels_number]
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])
    tr_index = []
    te_index = []
    train_x = np.empty((sizetr))
    train_y = np.empty((tr_size), dtype=int)
    test_x = np.empty((sizete))
    test_y = np.empty((te_size), dtype=int)
    trcont = 0
    tecont = 0
    class_number = 0
    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        indexes_cl = indexes[labels == cl]
        pixels_cl, labels_cl, indexes_cl = random_unison(pixels_cl, labels_cl, indexes_cl, rstate=rand_state)
        for cont, (a, b, c) in enumerate(zip(pixels_cl, labels_cl, indexes_cl)):
            if cont < train_set_size[class_number]:
                train_x[trcont, :, :, :] = a
                train_y[trcont] = b
                tr_index.append(c)
                trcont += 1
            else:
                test_x[tecont, :, :, :] = a
                test_y[tecont] = b
                te_index.append(c)
                tecont += 1
        class_number = class_number + 1
    tr_index = np.asarray(tr_index)
    te_index = np.asarray(te_index)
    train_x, train_y, tr_index = random_unison(train_x, train_y, tr_index, rstate=rand_state)
    return train_x, test_x, train_y, test_y, tr_index, te_index

# 其他小工具
# 创建路径
def setDir():
    filepath = 'dataset/split_dataset'
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
# 顺序随机打乱
def random_unison(a, b, c, rstate):
    assert len(a) == len(b) & len(a) == len(c)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p], c[p]

if __name__ == '__main__':

    # 加载配置
    with open("configs/config.yml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # 设置路径并清空缓存数据
    setDir()
    fix_data_path = 'dataset/split_dataset/'
    if os.path.exists(fix_data_path):
        shutil.rmtree(fix_data_path)
    os.makedirs(fix_data_path, exist_ok=True)
    if cfg['Root_path'] == 0:
        root_path = ''
    else:
        root_path = cfg['Root_path']
    cache_dir = root_path + 'dataset/data_cache_pool'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print("Cache directory has been cleared and is ready for use.")
    # 加载数据
    device = torch.device("cpu")
    data_path = 'dataset/'
    Project = cfg['Project']
    data = data_path + cfg['Data']['data']
    modes_number = string_to_int_list(cfg['Data']['modes_number'])
    band_number = sum(modes_number)

    datasets = gdal.Open(data, gdal.GA_ReadOnly)
    #band_number = datasets.RasterCount
    width = datasets.RasterXSize
    height = datasets.RasterYSize
    x_data = np.empty((height, width, band_number), dtype=np.float32)
    for band_idx in range(band_number):
        band = datasets.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        x_data[:, :, band_idx] = band.ReadAsArray()
    print("x shape is {} ".format(x_data.shape))

    # 归一化
    cumulative_modes = np.cumsum(modes_number)
    for i in range(len(modes_number)):
        if string_to_int_list(cfg['Preprocessing']['normalization'])[i] == 1:
            if i == 0:
                x_data[:, :, :modes_number[i]] = normalizations(x_data[:, :, :modes_number[i]])
            else:
                x_data[:, :, cumulative_modes[i-1]:cumulative_modes[i]] = normalizations(x_data[:, :, cumulative_modes[i-1]:cumulative_modes[i]])#数组前后切片，取前不取后（0为第一个索引号），单个切片为0开始索引。无引号为0开始为第1波段。
    # Standard Scaler处理
    if cfg['Preprocessing']['Standard_Scaler'] == 1:
        shapeor = x_data.shape
        x_data = x_data.reshape(-1, x_data.shape[-1])
        x_data = StandardScaler().fit_transform(x_data)
        x_data = x_data.reshape(shapeor)
    # PCA
    if cfg['Preprocessing']['PCA'] > 0:
        shapeor = x_data.shape
        x_data = x_data.reshape(-1, x_data.shape[-1])
        x_data = PCA(n_components=cfg['Preprocessing']['PCA']).fit_transform(x_data)
        shapeor = np.array(shapeor)
        shapeor[-1] = cfg['Preprocessing']['PCA']
        x_data = StandardScaler().fit_transform(x_data) #StandardScaler 将数据转换为均值为 0、标准差为 1 的标准正态分布。
        x_data = x_data.reshape(shapeor)
        print("PCA_x_data.shape:",x_data.shape)

    # 读取标签
    y = data_path + cfg['Data']['y']
    y = gdal.Open(y, gdal.GA_ReadOnly)
    num_bands = y.RasterCount
    width = y.RasterXSize
    height = y.RasterYSize
    y_data = np.empty((height, width, num_bands), dtype=np.float32)
    for band_idx in range(num_bands):
        band = y.GetRasterBand(band_idx + 1)
        y_data[:, :, band_idx] = band.ReadAsArray()
    print("y shape is {} ".format(y_data.shape))
    y_data = y_data.astype('uint8')
    num_class = len(np.unique(y_data)) - 1
    # 统计数据量
    x_train_number = []
    y_train_number = []
    x_val_number = []
    y_val_number = []
    x_test_number = []
    y_test_number = []
    # 制作PP并分割数据集
    if cfg['Preprocessing']['manual_segmentation'] == 0:
        # 生成PP
        file_names, patchesLocations_batch = create_PP(cfg, x_data, y_data)
        # 数据集分割
        for i, filename in enumerate(file_names):
            # 加载当前批次数据
            data = sio.loadmat(filename)
            patchesData = np.array(data['patchesData'])  # 显式转换为 numpy 数组
            patchesLabels = np.array(data['patchesLabels'])
            patchesLabels = patchesLabels.flatten().astype(np.int32)# 显式转换为 numpy 数组
            current_locations = [loc for loc in patchesLocations_batch if loc[2] == os.path.basename(filename)]
            current_locations = np.array(current_locations)[:, :2]  # 去除文件名信息
            # 对当前批次执行数据分割(分出训练集)
            x_train, x_test, y_train, y_test, train_idx, test_idx = split_data(
                patchesData, patchesLabels, current_locations,
                cfg['Preprocessing']["train_val_test_percent"][0],
                cfg["Random_seed"]
            )
            # 对当前批次执行数据分割(分出验证集与测试集)
            x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(
                x_test, y_test, test_idx,
                cfg['Preprocessing']["train_val_test_percent"][1],
                cfg["Random_seed"])
            # 保存训练集
            train_fname_npy = os.path.join(fix_data_path, f"x_train_batch{i}.npy")
            np.save(train_fname_npy, x_train)
            # 保存验证集
            val_fname_npy = os.path.join(fix_data_path, f"x_val_batch{i}.npy")
            np.save(val_fname_npy, x_val)
            # 保存测试集
            test_fname_npy = os.path.join(fix_data_path, f"x_test_batch{i}.npy")
            np.save(test_fname_npy, x_test)
            # 单独保存y数据
            np.save(os.path.join(fix_data_path, f"y_train_batch{i}.npy"), y_train)
            np.save(os.path.join(fix_data_path, f"y_val_batch{i}.npy"), y_val)
            np.save(os.path.join(fix_data_path, f"y_test_batch{i}.npy"), y_test)
            # 记录位置
            test_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
            for pos in new_test_index:
                row, col = int(pos[0]), int(pos[1]) #这里行列号与Arc gis中的显示倒置的
                test_positions[row, col] = 1
            np.save(pjoin(fix_data_path + f"testSet_position_batch{i}.npy"), test_positions)
            val_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
            for pos in val_index:
                row, col = int(pos[0]), int(pos[1]) #这里行列号与Arc gis中的显示倒置的
                val_positions[row, col] = 1
            np.save(pjoin(fix_data_path + f"valSet_positions_batch{i}.npy"), val_positions)
            train_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
            for pos in train_idx:
                row, col = int(pos[0]), int(pos[1]) #这里行列号与Arc gis中的显示倒置的
                train_positions[row, col] = 1
            np.save(pjoin(fix_data_path + f"trainSet_position_batch{i}.npy"), train_positions)
            # 信息打印
            print(f"x_train_batch{i} shape:", x_train.shape)
            print(f"y_train_batch{i} shape:", y_train.shape)
            print(f"x_val_batch{i} shape:", x_val.shape)
            print(f"y_val_batch{i} shape:", y_val.shape)
            print(f"x_test_batch{i} shape:", x_test.shape)
            print(f"y_test_batch{i} shape:", y_test.shape)
            print(f"testSet_position_batch{i} shape:", test_positions.shape)
            x_train_number.append(x_train.shape[0])
            y_train_number.append(y_train.shape[0])
            x_val_number.append(x_val.shape[0])
            y_val_number.append(y_val.shape[0])
            x_test_number.append(x_test.shape[0])
            y_test_number.append(y_test.shape[0])

    else:
        # 手动数据集分割
        # 加载训练集
        y_train_data = gdal.Open(os.path.join(data_path, cfg['Preprocessing']['y_train_tif']), gdal.GA_ReadOnly)
        num_bands = y_train_data.RasterCount
        width = y_train_data.RasterXSize
        height = y_train_data.RasterYSize
        y_train = np.empty((height, width, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            band = y_train_data.GetRasterBand(band_idx + 1)
            y_train[:, :, band_idx] = band.ReadAsArray()
        print("y_train shape is {} ".format(y_train.shape))
        y_train = y_train.astype('uint8')
        # 加载验证集与测试集
        y_test_data = gdal.Open(os.path.join(data_path, cfg['Preprocessing']['y_test_tif']), gdal.GA_ReadOnly)
        num_bands = y_test_data.RasterCount
        width = y_test_data.RasterXSize
        height = y_test_data.RasterYSize
        y_test = np.empty((height, width, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            band = y_test_data.GetRasterBand(band_idx + 1)
            y_test[:, :, band_idx] = band.ReadAsArray()
        print("y_test shape is {} ".format(y_test.shape))
        y_test = y_test.astype('uint8')
        # 数据集集分割掩码：y_train 中非零的区域
        train_mask = (y_train > 0).any(axis=2)  # 如果标签有多个波段，取任意波段非零的区域
        test_mask = (y_test > 0).any(axis=2)
        train_coords = np.argwhere(train_mask)  # 形状为 (N, 2)，每个元素是 (row, col)
        test_coords = np.argwhere(test_mask)  # 形状为 (M, 2)
        # 转换为集合以提高查找效率
        train_coords = set(tuple(coord) for coord in train_coords)
        test_coords = set(tuple(coord) for coord in test_coords)
        # 对x_data进行掩码处理
        x_train_masked = np.zeros_like(x_data)
        x_test_masked = np.zeros_like(x_data)
        for row, col in train_coords:
            x_train_masked[row, col] = x_data[row, col]
        for row, col in test_coords:
            x_test_masked[row, col] = x_data[row, col]

        # 分别对掩码后的训练集和测试集执行create_PP函数
        # 训练集保存
        file_names_train, patches_locations_batch_train = create_PP(cfg, x_train_masked, y_train)
        for i, filename in enumerate(file_names_train):
            # 加载当前批次数据
            data = sio.loadmat(filename)
            x_train = np.array(data['patchesData'])  # 显式转换为 numpy 数组
            y_train = np.array(data['patchesLabels'])
            y_train = y_train.flatten().astype(np.int32)# 显式转换为 numpy 数组
            train_idx = [loc for loc in patches_locations_batch_train if loc[2] == os.path.basename(filename)]
            train_idx = np.array(train_idx)[:, :2]  # 去除文件名信息
            # 保存训练集
            train_fname_npy = os.path.join(fix_data_path, f"x_train_batch{i}.npy")
            np.save(train_fname_npy, x_train)
            np.save(os.path.join(fix_data_path, f"y_train_batch{i}.npy"), y_train)
            print(f"x_train_batch{i} shape:", x_train.shape)
            print(f"y_train_batch{i} shape:", y_train.shape)
            x_train_number.append(x_train.shape[0])
            y_train_number.append(y_train.shape[0])
            train_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
            # 记录训练集位置
            for pos in train_idx:
                row, col = int(pos[0]), int(pos[1]) #这里行列号与Arc gis中的显示倒置的
                train_positions[row, col] = 1
            np.save(pjoin(fix_data_path + f"trainSet_position_batch{i}.npy"), train_positions)
        # 清除缓存
        cache_dir = root_path + 'dataset/data_cache_pool'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        # 测试集验证集保存
        file_names_test, patches_locations_batch_test = create_PP(cfg, x_test_masked, y_test)
        # 数据集分割
        for i, filename in enumerate(file_names_test):
            # 加载当前批次数据
            data = sio.loadmat(filename)
            patchesData = np.array(data['patchesData'])  # 显式转换为 numpy 数组
            patchesLabels = np.array(data['patchesLabels'])
            patchesLabels = patchesLabels.flatten().astype(np.int32)# 显式转换为 numpy 数组
            current_locations = [loc for loc in patches_locations_batch_test if loc[2] == os.path.basename(filename)]
            current_locations = np.array(current_locations)[:, :2]  # 去除文件名信息
            # 对当前批次执行数据分割(分出验证集与测试集)
            x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(
                patchesData, patchesLabels, current_locations,
                cfg['Preprocessing']["train_val_test_percent"][1],
                cfg["Random_seed"]
            )
            # 保存验证集
            val_fname_npy = os.path.join(fix_data_path, f"x_val_batch{i}.npy")
            np.save(val_fname_npy, x_val)
            # 保存测试集
            test_fname_npy = os.path.join(fix_data_path, f"x_test_batch{i}.npy")
            np.save(test_fname_npy, x_test)
            # 单独保存y数据
            np.save(os.path.join(fix_data_path, f"y_val_batch{i}.npy"), y_val)
            np.save(os.path.join(fix_data_path, f"y_test_batch{i}.npy"), y_test)
            # 记录测试集验证集位置
            test_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
            for pos in new_test_index:
                row, col = int(pos[0]), int(pos[1]) #这里行列号与Arc gis中的显示倒置的
                test_positions[row, col] = 1
            np.save(pjoin(fix_data_path + f"testSet_position_batch{i}.npy"), test_positions)
            val_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
            for pos in val_index:
                row, col = int(pos[0]), int(pos[1]) #这里行列号与Arc gis中的显示倒置的
                val_positions[row, col] = 1
            np.save(pjoin(fix_data_path + f"valSet_positions_batch{i}.npy"), val_positions)
            print(f"x_val_batch{i} shape:", x_val.shape)
            print(f"y_val_batch{i} shape:", y_val.shape)
            print(f"x_test_batch{i} shape:", x_test.shape)
            print(f"y_test_batch{i} shape:", y_test.shape)
            print(f"testSet_position_batch{i} shape:", test_positions.shape)
            x_val_number.append(x_val.shape[0])
            y_val_number.append(y_val.shape[0])
            x_test_number.append(x_test.shape[0])
            y_test_number.append(y_test.shape[0])

    print(f"x_train_batch number:", sum(x_train_number))
    print(f"y_train_batch number:", sum(y_train_number))
    print(f"x_val_batch number:", sum(x_val_number))
    print(f"y_val_batch number:", sum(y_val_number))
    print(f"x_test_batch number:", sum(x_test_number))
    print(f"y_test_batch number:", sum(y_test_number))
    print("creat dataset over!")

