"""
Evaluating inference time (whole HSI) and classification accuracy (test set) of TPPP-Nets
"""
import os
import scipy.io as sio
import torch
import argparse
import numpy as np
import yaml
import time
import auxil
from TPPI.utils import convert_state_dict
from TPPI.models import get_model
from sklearn.preprocessing import StandardScaler
from osgeo import gdal

# 常用工具
# 字符串转整型
def string_to_int_list(string):
    str_list = string.split(',')
    int_list = []
    for item in str_list:
        if item.lower() == "true":
            int_list.append(1)
        elif item.lower() == "false":
            int_list.append(0)
        else:
            int_list.append(int(item))
    return int_list
# 归一化
def normalizations(data):
    for i in range(data.shape[2]):
        min_value = np.min(data[:,:,i][data[:,:,i] > 0])
        max_value = np.max(data[:,:,i][data[:,:,i] > 0])
        data[:,:,i][data[:,:,i] > 0] = (data[:,:,i][data[:,:,i] > 0] - min_value) / (max_value - min_value)
    return data

# 核心函数
# 预测函数
def predict_patches(data, model, cfg, device, logdir):
    # 加载参数
    transfer_data_start = time.time()
    started = cfg["Data"]["band_selection"][0]
    end = cfg["Data"]["band_selection"][1]
    if cfg["Model"] == 'Shallow_Network':
        data2 = data[:,started:end,:,:]
        sklearn_data = data2.permute(0, 2, 3, 1)
        sklearn_data = sklearn_data.numpy()
        sklearn_data = sklearn_data.reshape((sklearn_data.shape[0], -1))
        # data2 = data[:, started:end, 2, 2]
        # sklearndata = data2.numpy()
    else:
        data = data.to(device)
    transfer_data_end = time.time()
    transfer_time = transfer_data_end - transfer_data_start
    predicted = []
    bs = cfg["Prediction"]["batch_size"]
    if cfg["Model"] == 'HRN':
        loaded_weights = np.loadtxt(os.path.join(logdir,"weights.csv"), delimiter=',')
    tsp = time.time()
    if cfg["Model"] == 'Shallow_Network':
        import joblib
        loaded_model = joblib.load(logdir+'/model.pkl')
        Probability = loaded_model.predict_proba(sklearn_data)
        print('Shallow Network predict')
    else:
        with torch.no_grad():
            for i in range(0, data.shape[0], bs):
                end_index = i + bs
                batch_data = data[i:end_index]
                batch_data = batch_data[:, started:end, :, :]
                if cfg["Model"] == 'HRN':
                    outputs, _1 = model(batch_data,loaded_weights)
                elif cfg["Model"] != 'HRN':
                    outputs = model(batch_data)
                [predicted.append(a) for a in outputs.cpu().numpy()]
        Probability = np.array(predicted)
    tep = time.time()
    prediction_time = tep - tsp
    return prediction_time, transfer_time, Probability
# 预测时间与精度计算
def timeCost_TPPP(cfg, logdir):
    # 参数准备
    name = cfg['Project']
    model_name = str(cfg['Model'])
    modes_number = string_to_int_list(cfg['Data']['modes_number'])
    in_channel = sum(modes_number)
    device = auxil.get_device()
    savepath = './Result/' + name + "_" + model_name + "_PPsize" + str(cfg['Preprocessing']['PP_size']) + "_epochs" + str(
        cfg['Train']['epochs']) + "_Channel" + str(in_channel)+'/'
    try:
        os.makedirs(savepath)
    except FileExistsError:
        pass
    data_path = 'dataset/'
    data = data_path + cfg['Data']['data']
    y = data_path + cfg['Data']['y']
    manually = cfg['Preprocessing']['manual_segmentation']
    band_number = sum(modes_number)

    # 加载数据
    datasets = gdal.Open(data, gdal.GA_ReadOnly)
    # band_number = datasets.RasterCount
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
                x_data[:, :, cumulative_modes[i - 1]:cumulative_modes[i]] = normalizations(x_data[:, :, cumulative_modes[i - 1]:cumulative_modes[i]])  # 数组切片，取后不取前。无引号为0开始为第一波段。
    # PCA
    if cfg['Preprocessing']['PCA'] > 0:
        from sklearn.decomposition import PCA
        shapeor = x_data.shape
        x_data = x_data.reshape(-1, x_data.shape[-1])
        x_data = PCA(n_components=cfg['Preprocessing']['PCA']).fit_transform(x_data)
        shapeor = np.array(shapeor)
        shapeor[-1] = cfg['Preprocessing']['PCA']
        x_data = StandardScaler().fit_transform(x_data)  # StandardScaler 将数据转换为均值为 0、标准差为 1 的标准正态分布。
        x_data = x_data.reshape(shapeor)
        print("PCA_x_data.shape:", x_data.shape)
    # Standard Scaler处理
    if cfg['Preprocessing']['Standard_Scaler'] == 1:
        shapeor = x_data.shape
        x_data = x_data.reshape(-1, x_data.shape[-1])
        x_data = StandardScaler().fit_transform(x_data)
        x_data = x_data.reshape(shapeor)
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

    # 预测与精度计算
    time_pre_start = time.time()
    # create patch
    file_names, patchesLocations_batch = auxil.create_PP(cfg, x_data, y_data)
    pre_list = []
    locations_list = []
    time_pre_processing_all = 0.000
    tts_all = 0.000
    ts_all = 0.000
    tt_all = 0.000
    pt_all = 0.000
    comb_all = 0.000
    for i, filename in enumerate(file_names):
        # 加载当前批次数据
        data = sio.loadmat(filename)
        x_data = np.array(data['patchesData'])  # 显式转换为 numpy 数组
        current_locations = [loc for loc in patchesLocations_batch if loc[2] == os.path.basename(filename)]
        current_locations = np.array(current_locations)[:, :2]  # 去除文件名信息
        locations_list.append(current_locations)
    # NHWC -> NCHW
        x_data = x_data.transpose(0, 3, 1, 2)
        x_data = torch.from_numpy(x_data).float()
        time_pre_end = time.time()
        time_pre_processing = time_pre_end - time_pre_start
        time_pre_processing_all = time_pre_processing_all + time_pre_processing
        print("creat patch {} data over!", x_data.shape)

        # setup model:
        model = get_model(cfg['Model'], cfg['Project'])
        if cfg["Model"] != 'Shallow_Network':
            state = convert_state_dict(
                torch.load(os.path.join(logdir, cfg["Train"]["best_model_path"]),weights_only=False)[
                    "model_state"])
            model.load_state_dict(state)
            model.eval()

        # transfer model to GPU
        ts1 = time.time()
        model.to(device)
        ts2 = time.time()

        # predicting
        print("predicting...")
        pt, tt, outputs = predict_patches(x_data, model, cfg, device, logdir)
        tts_all = tts_all + (tt - (ts2 - ts1))
        ts_all = ts_all + (ts2 - ts1)
        tt_all = tt_all + tt
        pt_all = pt_all + pt

        # get result and reshape
        comb_s = time.time()
        outputs = np.array(outputs)
        pred = np.argmax(outputs, axis=1)
        comb_e = time.time()
        comb_all = comb_all + (comb_e - comb_s)
        pred += 1 # 标签值从1开始
        pre_list.append(pred)

    # show predicted result
    # 分批显示方法
    pred_full = np.full((y_data.shape[0], y_data.shape[1]), fill_value=-1, dtype=np.int64)
    for i in range(len(pre_list)):
        pred_batch = pre_list[i]
        locations_batch = locations_list[i]
        for j in range(len(pred_batch)):
            r, c = locations_batch[j]
            r = int(r)
            c = int(c)
            pred_full[r, c] = pred_batch[j]
    # 保存为带坐标信息的图片（全部图片）
    gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")
    geotransform = y.GetGeoTransform()
    projection = y.GetProjection()
    output_path = savepath + name + "_" + model_name + "_predictions_All.tif"
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        output_path,
        pred_full.shape[1],  # Width
        pred_full.shape[0],  # Height
        1,  # Number of bands
        gdal.GDT_Byte  # Data type (change if needed)
    )
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(pred_full.astype(np.uint8))
    output_band.SetScale(1.0)  # 每个单位代表 1.0
    output_band.SetOffset(0.0)  # 没有偏移
    output_band.FlushCache()
    output_dataset = None
    auxil.decode_segmap(pred_full)

    # 计算精度
    final_position = np.zeros_like(y_data[..., 0], dtype=np.int8)
    for testSet_position_i in range(cfg["Preprocessing"]["big_data_batch"]):
        teposition_path = 'dataset/split_dataset/' + 'testSet_position_batch' + str(testSet_position_i) + '.npy'
        position = np.load(teposition_path)
        final_position = np.maximum(final_position, position)
    np.save('dataset/split_dataset/Pre_test_position.npy', final_position)
    prednew = pred_full[final_position[:,:] == 1]
    gtnew = y_data[final_position[:,:] == 1]
    classification, confusion, result = auxil.reports(prednew, gtnew)
    result_info = "OA AA Kappa and each Acc:\n" + str(result)

    # 保存预测结果
    import spectral
    spectral.save_rgb(savepath + name + "_" + model_name + "_predictions_All.jpg", pred_full.astype(int),
                      colors=spectral.spy_colors)
    mask = np.zeros((y_data.shape[0],y_data.shape[1]), dtype='bool')
    mask[y_data[:,:,0] == 0] = True
    pred_full[mask] = 0
    spectral.save_rgb(savepath + name +"_"+ model_name + "_predictions_GT.jpg", pred_full.astype(int),
                      colors=spectral.spy_colors)
    # 保存为带坐标信息的图片（标签图片）
    gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")
    geotransform = y.GetGeoTransform()
    projection = y.GetProjection()
    output_path = savepath + name + "_" + model_name + "_predictions_GT.tif"
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        output_path,
        pred_full.shape[1],  # Width
        pred_full.shape[0],  # Height
        1,  # Number of bands
        gdal.GDT_Byte  # Data type (change if needed)
    )
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(pred_full.astype(np.uint8))
    output_band.SetScale(1.0)  # 每个单位代表 1.0
    output_band.SetOffset(0.0)  # 没有偏移
    output_band.FlushCache()
    output_dataset = None
    auxil.decode_segmap(pred_full)

    # report time cost and accuracy
    print("******************** Time ***********************")
    print("Data_processing time is:", time_pre_processing_all) # 数据预处理时间
    print("Transfer time is:", tts_all, "  model:", ts_all, "  data:", tt_all) # 1.装载总时间。2.模型装载时间。3.数据装载时间
    print("Prediction time is:", pt_all) # 真实预测时间（论文使用）
    print("combine time is:", comb_all) # get result and reshape的时间
    print('Total inference time is:', time_pre_processing_all + tt_all + ts_all +pt_all +comb_all) # 总花费时间

    # report classification accuracy
    print("****************** Accuracy *********************")
    print(result_info)
    print("****************** classification *********************")
    print(str(classification))
    print("****************** confusion *********************")
    print(str(confusion))

    print("\n")
    file_name = savepath + "classification_report_" + name +"_"+ model_name +"dataset.txt"
    with open(file_name, 'w') as x_file:
        x_file.write("******************** Time ***********************")
        x_file.write('\n')
        x_file.write("Data_processing time is:{}".format(time_pre_processing))
        x_file.write('\n')
        x_file.write("Transfer time is:{}".format(tt + (ts2 - ts1), "  model:", ts2 - ts1, "  data:", tt))
        x_file.write('\n')
        x_file.write("Prediction time is:{}".format(pt))
        x_file.write('\n')
        x_file.write("combine time is:{}".format(comb_e - comb_s))
        x_file.write('\n')
        x_file.write('Total inference time is:{}'.format(time_pre_processing + tt + (ts2 - ts1) + pt + comb_e - comb_s))
        x_file.write('\n')
        # report classification accuracy
        x_file.write("****************** Accuracy *********************")
        x_file.write('\n')
        x_file.write('{}'.format(result_info))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write("****************** classification *********************")
        x_file.write('\n')
        x_file.write('{}'.format(str(classification)))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write("****************** confusion *********************")
        x_file.write('\n')
        x_file.write('{}'.format(str(confusion)))

def Run_Predict():
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

    name = cfg['Project']
    model_name= str(cfg['Model'])
    modes_number = string_to_int_list(cfg['Data']['modes_number'])
    in_channel = sum(modes_number)

    logdir = './Result/' + name + "_" + model_name + "_PPsize" + str(cfg['Preprocessing']['PP_size']) + "_epochs" + str(
        cfg['Train']['epochs']) + "_Channel" + str(in_channel) + '/'+str(cfg["Run_ID"])
    timeCost_TPPP(cfg, logdir)

if __name__ == "__main__":
    Run_Predict()
