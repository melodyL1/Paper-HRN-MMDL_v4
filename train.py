import time
import warnings
import sys
import os
import yaml
import shutil
import random
import argparse

import auxil
from tensorboardX import SummaryWriter
import torch.nn.parallel
from TPPI.models import get_model
from TPPI.optimizers import get_optimizer
from TPPI.schedulers import get_scheduler
from TPPI.loaders.Dataloader_train import *
from TPPI.utils import get_logger
from TPPP_predict import Run_Predict
import subprocess
from torchsummary import summary
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
warnings.filterwarnings(action='ignore')

# 常用工具函数
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

# 训练函数
def train(cfg, train_files, val_files, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger):
    # tr_writer：训练时 TensorBoard 的写入器；logdir: TensorBoard 的日志文件目录。
    # logger: 日志记录器。
    # TODO 1
    if cfg["Model"] == 'Shallow_Network':   # 浅层网络
        all_x_Shallow_Network = []
        all_y_Shallow_Network = []
        for train_file in train_files:
            train_loader, x_Shallow_Network, y_Shallow_Network = get_dataLoader(train_file, cfg, logdir)
            all_x_Shallow_Network.append(x_Shallow_Network)
            all_y_Shallow_Network.append(y_Shallow_Network)
        # 合并所有特征和标签
        x_Shallow_Network = np.concatenate(all_x_Shallow_Network, axis=0)
        y_Shallow_Network = np.concatenate(all_y_Shallow_Network, axis=0)
        import joblib
        if cfg["Model_detail"]["Shallow_Network"] == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            Shallow_Network_classifier = RandomForestClassifier()
            Shallow_Network_classifier.fit(x_Shallow_Network, y_Shallow_Network)
            joblib.dump(Shallow_Network_classifier, logdir + '/model.pkl')
        print('Sklearn Finish')

    else:        # 深度学习
        save_epoch = []
        start_epoch = 0
        continue_path = os.path.join(logdir, "continue_model.pkl")#表示一个模型的保存路径，在训练过程中可以从该模型继续训练。
        if os.path.isfile(continue_path):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(continue_path)#从检查点加载模型和优化器
            )
            checkpoint = torch.load(continue_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    continue_path, checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(continue_path))
        # 设置初始化参数
        best_acc = -1
        epoch = start_epoch
        flag = True
        classs = cfg["Data"]["class"]
        # 多模态参数
        if cfg["Model"] == 'HRN':
            all_targets0 = []
            all_pred = [[] for _ in range(len(modes_number) + cfg['Model_detail']['Network_Dilated'])]
            weights = np.ones((len(modes_number) + cfg['Model_detail']['Network_Dilated'], classs))

        # 开始训练
        while epoch <= cfg["Train"]["epochs"] and flag:
            model.train()
            # TODO 2
            if cfg["Model"] == 'HRN':
                all_losses = []
                all_accs = []
                for train_file in train_files:
                    train_loader, x_Shallow_Network, y_Shallow_Network = get_dataLoader(train_file, cfg, logdir)
                    train_accs = np.ones((len(train_loader))) * -1000.0
                    train_losses = np.ones((len(train_loader))) * -1000.0
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets) # 将输入和目标张量转换为autograd.Variable类型。
                        weights = weights
                        outputs, out = model(inputs, weights)
                        targets0 = targets.cpu().detach().numpy()
                        all_targets0.append(targets0)
                        for out_idx in range(len(out)):
                            out[out_idx] = out[out_idx].cpu().detach().numpy()
                            pred = np.argmax(out[out_idx], axis=1)
                            all_pred[out_idx].append(pred)
                        loss = loss_fn(outputs, targets)
                        train_losses[batch_idx] = loss.item() # 记录该batch的训练损失。
                        train_accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item() # 记录该batch的训练准确率
                        all_losses.append(train_losses[batch_idx])
                        all_accs.append(train_accs[batch_idx])
                        optimizer.zero_grad() # 清空优化器的梯度缓存。
                        loss.backward() # 进行反向传播计算梯度。
                        optimizer.step() # 根据梯度更新模型参数。

                # TODO 核心权重计算
                all_preds = [np.concatenate(pred_weights, axis=0) for pred_weights in all_pred]
                all_targets01 = np.concatenate(all_targets0, axis=0)
                all_targets01 = all_targets01.reshape(-1, 1)
                all_targets02 = all_targets01.reshape(-1, )
                factor_weights = []
                for pred in all_preds:
                    classification, confusion, reports_result = auxil.reports(pred, all_targets01)
                    recall_weights = reports_result[3:3 + classs]
                    f1_weights = fbeta_score(all_targets02, pred, beta=1, average=None) # F1作为核心权重因子
                    f1_weights = f1_weights.tolist()
                    factor_weights.append(recall_weights)
                results_weights_numpy = [np.array(res) for res in  factor_weights]
                weights = np.vstack([
                    res / np.sum(results_weights_numpy, axis=0)
                    for res in results_weights_numpy
                ])
                print(weights)

                # 训练后续过程
                scheduler.step()  # 更新学习率调度程序。
                train_loss = np.mean(all_losses)
                train_acc = np.mean(all_accs)
                fmt_str = "Iter [{:d}/{:d}]  \nTrain_loss: {:f}  Train_acc: {:f}"
                print_str = fmt_str.format(
                    epoch + 1,
                    cfg["Train"]["epochs"],
                    train_loss,
                    train_acc,
                )
                tr_writer.add_scalar("loss", train_loss, epoch + 1)  # 使用tensorboard记录训练损失。
                print(print_str)
                logger.info(print_str)
                state = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }  # 状态保存在state字典中
                # save to the continue path
                torch.save(state, continue_path)
                epoch += 1
                if (epoch + 1) % cfg["Train"]["val_interval"] == 0 or (epoch + 1) == cfg["Train"]["epochs"]:
                    model.eval()
                    val_loss_all = []
                    val_acc_all = []
                    for val_file in val_files:
                        val_loader, x_Shallow_Network, y_Shallow_Network = get_dataLoader(val_file, cfg, logdir)
                        val_accs = np.ones((len(val_loader))) * -1000.0
                        val_losses = np.ones((len(val_loader))) * -1000.0
                        with torch.no_grad():
                            for batch_idy, (inputs, targets) in enumerate(val_loader):
                                inputs, targets = inputs.to(device), targets.to(device)
                                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                                outputs, _1 = model(inputs, weights)
                                val_losses[batch_idy] = loss_fn(outputs, targets).item()
                                val_accs[batch_idy] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
                            val_loss_all.extend(val_losses.tolist())
                            val_acc_all.extend(val_accs.tolist())
                    val_loss = np.mean(val_loss_all)
                    val_acc = np.mean(val_acc_all)
                    fmt_str = "Val_loss: {:f}  Val_acc: {:f}"
                    print_str = fmt_str.format(
                        val_loss,
                        val_acc,
                    )
                    val_writer.add_scalar("loss", val_loss, epoch)
                    print(print_str)
                    logger.info(print_str)

                    if val_acc > best_acc:
                        save_epoch.append(epoch)
                        best_acc = val_acc
                        state = {
                            'epoch': epoch + 1,
                            'best_acc': best_acc,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                        }
                        torch.save(state, os.path.join(logdir, "best_model.pth.tar"))
                        np.savetxt(os.path.join(logdir, "weights.csv"), weights, delimiter=',') # 保存权重信息
                if epoch == cfg["Train"]["epochs"]:
                    flag = False
                    break
                # TODO 3
            else:
                all_losses = []
                all_accs = []
                for train_file in train_files:
                    train_loader, x_Shallow_Network, y_Shallow_Network = get_dataLoader(train_file, cfg, logdir)
                    train_accs = np.ones((len(train_loader))) * -1000.0
                    train_losses = np.ones((len(train_loader))) * -1000.0
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                        outputs = model(inputs)
                        if len(outputs.shape) == 2:
                            if outputs.shape[1] != cfg['Data']['class']:
                                raise ValueError(f"模型输出类别个数异常! ")
                            else:
                                loss = loss_fn(outputs, targets)
                        else:
                            if outputs.shape[0] != cfg['Data']['class']:
                                raise ValueError(f"模型输出类别个数异常! ")
                            else:
                                outputs = outputs.unsqueeze(0)
                                loss = loss_fn(outputs, targets)
                        train_losses[batch_idx] = loss.item()  # 记录该batch的训练损失。
                        train_accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()  # 记录该batch的训练准确率
                        all_losses.append(train_losses[batch_idx])
                        all_accs.append(train_accs[batch_idx])
                        optimizer.zero_grad()  # 清空优化器的梯度缓存。
                        loss.backward()  # 进行反向传播计算梯度。
                        optimizer.step()  # 根据梯度更新模型参数。
                scheduler.step()  # 更新学习率调度程序。
                train_loss = np.mean(all_losses)
                train_acc = np.mean(all_accs)
                fmt_str = "Iter [{:d}/{:d}]  \nTrain_loss: {:f}  Train_acc: {:f}"
                print_str = fmt_str.format(
                    epoch + 1,
                    cfg["Train"]["epochs"],
                    train_loss,
                    train_acc,
                )
                tr_writer.add_scalar("loss", train_loss, epoch + 1)  # 使用tensorboard记录训练损失。
                print(print_str)
                logger.info(print_str)
                state = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }  # 状态保存在state字典中
                # save to the continue path
                torch.save(state, continue_path)
                epoch += 1
                if (epoch + 1) % cfg["Train"]["val_interval"] == 0 or (epoch + 1) == cfg["Train"]["epochs"]:
                    model.eval()
                    val_loss_all = []
                    val_acc_all = []
                    for val_file in val_files:
                        val_loader, x_Shallow_Network, y_Shallow_Network = get_dataLoader(val_file, cfg, logdir)
                        val_accs = np.ones((len(val_loader))) * -1000.0
                        val_losses = np.ones((len(val_loader))) * -1000.0
                        with torch.no_grad():
                            for batch_idy, (inputs, targets) in enumerate(val_loader):
                                inputs, targets = inputs.to(device), targets.to(device)
                                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                                outputs = model(inputs)
                                val_losses[batch_idy] = loss_fn(outputs, targets).item()
                                val_accs[batch_idy] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[
                                    0].item()
                            val_loss_all.extend(val_losses.tolist())
                            val_acc_all.extend(val_accs.tolist())
                    val_loss = np.mean(val_loss_all)
                    val_acc = np.mean(val_acc_all)
                    fmt_str = "Val_loss: {:f}  Val_acc: {:f}"
                    print_str = fmt_str.format(
                        val_loss,
                        val_acc,
                    )
                    val_writer.add_scalar("loss", val_loss, epoch)
                    print(print_str)
                    logger.info(print_str)
                    if val_acc > best_acc:
                        save_epoch.append(epoch)
                        print(save_epoch)
                        best_acc = val_acc
                        state = {
                            'epoch': epoch + 1,
                            'best_acc': best_acc,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                        }
                        torch.save(state, os.path.join(logdir, "best_model.pth.tar"))
                if epoch == cfg["Train"]["epochs"]:
                    flag = False
                    break

if __name__ == '__main__':
    # 加载参数
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    name = cfg['Project']
    Random_seed = cfg['Random_seed']
    model_name= str(cfg['Model'])
    modes_number = string_to_int_list(cfg['Data']['modes_number'])
    in_channel = sum(modes_number)
    band_selection_in_channel = cfg["Data"]["band_selection"][1] - cfg["Data"]["band_selection"][0]
    device = auxil.get_device()

    # 创建目录与日志
    logdir = './Result/' + name + "_" + model_name + "_PPsize" + str(cfg['Preprocessing']['PP_size']) + "_epochs" + str(
        cfg['Train']['epochs']) + "_Channel" + str(in_channel) + '/'+str(cfg["Run_ID"])
    logs_file = os.path.join(logdir,"logs_file.txt")
    if os.path.exists(logdir):
        import shutil
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # #TODO logger
    tr_writer = SummaryWriter(log_dir=os.path.join(logdir+"/train/"))
    val_writer = SummaryWriter(log_dir=os.path.join(logdir+"/val/"))
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let begin!")

    # Setup seeds
    torch.manual_seed(cfg.get("seed", Random_seed))
    torch.cuda.manual_seed(cfg.get("seed", Random_seed))
    np.random.seed(cfg.get("seed", Random_seed))
    random.seed(cfg.get("seed", Random_seed))

    # Setup device
    device = auxil.get_device()

    # Setup Dataloader
    train_files, val_files, num_classes, n_bands = get_trainLoader_list(cfg,logdir)

    # Setup Model
    model = get_model(cfg['Model'], cfg['Project'])
    model = model.to(device)

    PPsize=cfg['Preprocessing']['PP_size']

    # summary(model, (n_bands,PPsize,PPsize))

    print("model load successfully")

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)  #sgd, adam, RMSprop
    optimizer_detail_params = {k: v for k, v in cfg["Train"]["optimizer"]["optimizer_detail"].items() if k != "optimizer_name"}
    optimizer_outer_params = {k: v for k, v in cfg["Train"]["optimizer"].items() if k not in ["optimizer_detail"]}
    optimizer_params = {**optimizer_outer_params, **optimizer_detail_params}
    if cfg["Train"]["optimizer"]["optimizer_detail"]["optimizer_name"] == "SGD":
        optimizer_params.pop("alpha", None)
        optimizer_params.pop("betas", None)
        print("optimizer = sgd")
    elif cfg["Train"]["optimizer"]["optimizer_detail"]["optimizer_name"] == "Adam":
        optimizer_params.pop("alpha", None)
        optimizer_params.pop("momentum", None)
        print("optimizer = adam")
    elif cfg["Train"]["optimizer"]["optimizer_detail"]["optimizer_name"] == "RMSprop":
        optimizer_params.pop("momentum", None)
        optimizer_params.pop("betas", None)
        print("optimizer = RMSprop")

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    # Setup lr_scheduler
    scheduler = get_scheduler(optimizer, cfg["Train"]["lr_schedule"])

    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # training model
    start_time = time.time()
    logger.info("Start_time {}".format(start_time))
    train(cfg, train_files, val_files, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger)
    end_time = time.time()
    logger.info("End_time {}".format(end_time))
    elapsed_time = end_time - start_time
    logger.info("Elapsed_time {}".format(elapsed_time))
    print("训练时间(秒)：", elapsed_time)

    # training over!
    print("Training is over!")
    logger.info("Training is over!")

    #RunPredict
    start_time = time.time()
    Run_Predict()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("总预测时间(秒)：", elapsed_time)
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    run_txt = './Result/' + name + "_" + model_name + "_PPsize" + str(cfg['Preprocessing']['PP_size']) + "_epochs" + str(
        cfg['Train']['epochs']) + "_Channel" + str(in_channel) + "/" + "classification_report_" + name + "_" + model_name + "dataset.txt"
    subprocess.run(["notepad.exe", run_txt])
