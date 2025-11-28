import os
import numpy as np
import rasterio

def merge_positions_to_tif(fix_data_path, output_path, reference_tif, big_data_batch):
    """
    将 trainSet_position_batch{i}.npy / valSet_position_batch{i}.npy / testSet_position_batch{i}.npy
    拼接还原为完整 TIF 文件，用于检查训练/验证/测试集分布是否有交集。

    参数:
        fix_data_path: str
            存放 batch 数据的目录
        output_path: str
            输出 TIF 文件目录
        reference_tif: str
            原始 y.tif 文件路径，用于获取空间参考
        big_data_batch: int
            分割 batch 的数量
    """
    os.makedirs(output_path, exist_ok=True)

    # 打开参考 TIF，读取空间信息
    with rasterio.open(reference_tif) as ref:
        meta = ref.meta.copy()

    # 修改 metadata 以便保存单波段 TIF
    meta.update(count=1, dtype="int32")

    # 初始化拼接矩阵
    train_full = None
    val_full = None
    test_full = None

    for i in range(big_data_batch):
        train_pos = np.load(os.path.join(fix_data_path, f"trainSet_position_batch{i}.npy")).astype(np.int32)
        val_pos = np.load(os.path.join(fix_data_path, f"valSet_positions_batch{i}.npy")).astype(np.int32)
        test_pos = np.load(os.path.join(fix_data_path, f"testSet_position_batch{i}.npy")).astype(np.int32)

        if train_full is None:
            train_full = np.zeros_like(train_pos, dtype=np.int32)
            val_full   = np.zeros_like(val_pos, dtype=np.int32)
            test_full  = np.zeros_like(test_pos, dtype=np.int32)

        # 按 batch 累加
        train_full += train_pos
        val_full   += val_pos
        test_full  += test_pos

    # 保存为 tif
    def save_tif(array, filename):
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(array.astype(np.int32), 1)

    save_tif(train_full, os.path.join(output_path, "train_positions_full.tif"))
    save_tif(val_full,   os.path.join(output_path, "val_positions_full.tif"))
    save_tif(test_full,  os.path.join(output_path, "test_positions_full.tif"))

    print("✅ 拼接完成！输出在:", output_path)
    print("检查：ArcGIS/QGIS 叠加 train/val/test_positions_full.tif，看看是否有重叠。")

    # 生成 Pre_test_position
    pre_test_npy_path = 'dataset/split_dataset/Pre_test_position.npy'
    output_tif_path = 'dataset/merged_positions/Pre_test_positions_full.tif'  # 输出路径
    # 读取预测的 npy 文件
    pre_test_position = np.load(pre_test_npy_path).astype(np.int32)


    # 写入 GeoTIFF 文件
    with rasterio.open(output_tif_path, 'w', **meta) as dst:
        dst.write(pre_test_position, 1)

    print(f"✅ 已成功生成带坐标信息的 GeoTIFF 文件: {output_tif_path}")
    print("可以在 ArcGIS / QGIS 中叠加查看预测区域分布。")


if __name__ == "__main__":
    fix_data_path = "dataset/split_dataset"  # 存放 batch 数据的路径
    output_path = "dataset/merged_positions" # 输出目录
    reference_tif = "dataset/y.tif"          # 原始 y.tif，用于空间参考
    big_data_batch = 5  # 按你的 cfg['Preprocessing']["big_data_batch"]

    merge_positions_to_tif(fix_data_path, output_path, reference_tif, big_data_batch)
