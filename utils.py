import datetime
import glob
import sys
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from ptflops import get_model_complexity_info
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from torch import optim
import torch
import os
import seaborn as sns
import pandas as pd
from torch.autograd import Variable
from dataset_preparation import Dataset_mat_MTL

from model.modelA_MTL import MTL_Net
from model.modelB_singleTask import Single_Task_Net
from model.modelC_multiClassifier import Multi_Classifier


class Logger(object):
    """
    A logger used to save console output to file
    Usage:
        just define :
            log1 = Logger('console output.log', path=save_dir)
            sys.stdout = log1
        before the "print()" function
    """

    def __init__(self, filename="Default.log", path="./"):
        self.terminal = sys.stdout
        self.log = ''
        self.filename = filename
        self.path = path

    def write(self, message):
        self.terminal.write(message)
        self.log += message

    def save(self):
        with open(self.path + '/' + self.filename, 'a') as f:
            f.write(self.log)

    def flush(self):
        pass


def draw_confusion_matrix(confusion_matrix, leibie1, font_scale=2.5, y_offset=0.5, title1=' ', is_show=False,
                          is_save=False, savepath='./', figsize=(7, 6.8)):
    sns.set(font="Times New Roman", font_scale=font_scale)
    # plt.figure(figsize=(8,8))
    f, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(confusion_matrix)
    df.columns = leibie1
    df.index = leibie1
    # sns.set(font="simhei")
    h = sns.heatmap(df, annot=True, ax=ax, cmap="OrRd", annot_kws={'size': 16, 'weight': 'bold'}, cbar=False,
                    fmt="d")
    # cb = h.figure.colorbar(h.collections[0])
    # cb.ax.tick_params(labelsize=16)

    plt.xticks(np.arange(y_offset, len(leibie1) + y_offset, 1), leibie1, fontsize=16, weight='bold')
    plt.yticks(np.arange(y_offset, len(leibie1) + y_offset, 1), leibie1, fontsize=16, weight='bold')

    ax.set_xlabel('Predicted Value', fontsize=16, weight='bold')
    ax.set_ylabel('True Value', fontsize=16, weight='bold')
    plt.title(title1)
    plt.tight_layout()
    plt.savefig(savepath)

    if is_show:
        plt.show()


def main_process(model_type='MTL', is_test=False, pth_file=None, GPU_device=True, dataset_ram=True,
                 random_state=1, fold_index=0, log_savedir='./', batch_size=32, epoch_num=40,
                 trainVal_set_striking='./',
                 trainVal_set_excavating='./',
                 test_set_striking='./',
                 test_set_excavating='./',
                 ):
    if model_type == 'MTL':
        note = 'model_type={} is_test={}'.format(model_type, is_test)
        model = MTL_Net()
    elif model_type == 'single_distance':
        note = 'model_type={} is_test={}'.format(model_type, is_test)
        model = Single_Task_Net(task='distance')
    elif model_type == 'single_event':
        note = 'model_type={} is_test={}'.format(model_type, is_test)
        model = Single_Task_Net(task='event')
    elif model_type == 'multi_classifier':
        note = 'model_type={} is_test={}'.format(model_type, is_test)
        model = Multi_Classifier()
    else:
        raise ValueError

    save_dir = log_savedir + '/{} {}/'.format(
        datetime.datetime.now().strftime("%m-%d-%H_%M_%S"), note)

    # Create the output file directory named date + note
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # backup the source files
    # if not os.path.exists(save_dir + 'src'):
    #     os.makedirs(save_dir + 'src')
    # py_file = glob.glob(r'./*.py')
    # for filename in py_file:
    #     copyfile(filename, save_dir + 'src/' + os.path.basename(filename))

    # Save the console output to file
    log1 = Logger('console output.log', path=save_dir)
    sys.stdout = log1
    print(__file__)

    if not torch.cuda.is_available():
        GPU_device = False

    if not pth_file == None:
        model.load_state_dict(torch.load(pth_file, map_location='cpu'), strict=True)
    if GPU_device == True:
        model.cuda()

    # # -Compute the model complexity
    # macs, params = get_model_complexity_info(model, (1, 100, 250), as_strings=False,
    #                                          print_per_layer_stat=False, verbose=True, ost=log1)
    # print('{:<30}  {:<8}'.format('MACs: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    base_lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)

    if model_type == 'multi_classifier':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.NLLLoss()

    if is_test:
        dataset_striking = test_set_striking
        dataset_excavating = test_set_excavating
    else:
        dataset_striking = trainVal_set_striking
        dataset_excavating = trainVal_set_excavating

    dataset_MTL = Dataset_mat_MTL(dataset_dir_striking=dataset_striking, dataset_dir_excavating=dataset_excavating,
                                  random_state=random_state, ram=dataset_ram, is_test=is_test, fold_index=fold_index,
                                  multi_categories=(True if model_type == 'multi_classifier' else False))

    dataloader1 = {}
    dataloader1['train'] = torch.utils.data.DataLoader(dataset_MTL.dataset['train'], batch_size=batch_size,
                                                       shuffle=True, num_workers=0)
    dataloader1['val'] = torch.utils.data.DataLoader(dataset_MTL.dataset['val'], batch_size=batch_size,
                                                     shuffle=True, num_workers=0)

    if model_type == 'MTL':
        trainer_MTL(model=model, data_loader=dataloader1, epoch_num=epoch_num + 1, start_epoch=0, optimizer=optimizer,
                    criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True, is_test=is_test)
    elif model_type == 'single_distance':
        trainer_single_task(model=model, data_loader=dataloader1, epoch_num=epoch_num + 1, start_epoch=0,
                            optimizer=optimizer,
                            criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True,
                            is_test=is_test, task='distance')
    elif model_type == 'single_event':
        trainer_single_task(model=model, data_loader=dataloader1, epoch_num=epoch_num + 1, start_epoch=0,
                            optimizer=optimizer,
                            criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True,
                            is_test=is_test, task='event')

    elif model_type == 'multi_classifier':
        trainer_multiClassifier(model=model, data_loader=dataloader1, epoch_num=epoch_num + 1, start_epoch=0,
                                optimizer=optimizer,
                                criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True,
                                is_test=is_test)
    else:
        raise ValueError

    # Save the training lines
    if not is_test:
        linelist = ['trainAccLine', 'trainLossLine',
                    'testAccLine', 'testLossLine']
        for linename in linelist:
            line = np.load(save_dir + linename + '.npy')
            plt.figure()

            if model_type == 'MTL':
                plt.plot(line[0], label='distance')
                plt.plot(line[1], label='event')
            elif model_type == 'single_distance' or model_type == 'single_event':
                plt.plot(line[0], label=model_type)
            elif model_type == 'multi_classifier':
                if linename == 'trainLossLine' or linename == 'testLossLine':
                    plt.plot(line[0], label=model_type)
                else:
                    plt.plot(line[0], label='distance')
                    plt.plot(line[1], label='event')
            else:
                raise ValueError

            plt.legend()
            plt.savefig(save_dir + linename + '.png')
            plt.close()


    # Save the confusion matrix
    else:
        leibie_event = ['Striking', 'Excavating ']
        leibei_distance = ['{}m'.format(i) for i in range(16)]
        cm_list = glob.glob(save_dir + 'confusion matrix*.npy')
        for cm in cm_list:
            mat1 = np.load(cm)
            if len(mat1[0]) == 16:
                draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibei_distance, figsize=(6.5, 6),
                                      savepath=save_dir + '/confusion matrix distance.svg')
            elif len(mat1[0]) == 2:
                draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibie_event, figsize=(4.5, 4),
                                      savepath=save_dir + '/confusion matrix event.svg')
            else:
                raise ValueError

    log1.save()


def trainer_MTL(model, epoch_num, start_epoch, optimizer, criterion, data_loader, save_dir,
                use_gpu=True, save_output=False, is_test=False):
    start_time = datetime.datetime.now()

    def adjust_lr(optimizer):
        for param in optimizer.param_groups:
            param["lr"] = param["lr"] / 1.5
        return optimizer

    LossLine = [[], []]
    AccLine = [[], []]
    testAccLine = [[], []]
    testLossLine = [[], []]
    acc_sum100 = [0, 0]
    loss_sum100 = [0, 0]

    for epoch in range(start_epoch, epoch_num):

        # Validation for every epoch
        if epoch % 5 == 0:

            optimizer = adjust_lr(optimizer)

            model.train(False)
            model.eval()
            ValAcc = [0, 0]
            ValBatch = [0, 0]
            testloss = [0, 0]

            pred_lst = [[], []]
            label_lst = [[], []]

            # t_predict = datetime.datetime.now()
            for val_cnt, val_data in enumerate(data_loader["val"]):

                valX, distance_label, event_label = val_data

                if use_gpu:
                    valX = Variable(valX.cuda())
                    distance_label = Variable(distance_label.cuda())
                    event_label = Variable(event_label.cuda())
                else:
                    valX = Variable(valX)
                    distance_label = Variable(distance_label)
                    event_label = Variable(event_label)

                out1, out2 = model(valX)

                l1 = criterion(out1, distance_label.long())
                l2 = criterion(out2, event_label.long())

                testloss[0] += l1.cpu().data
                testloss[1] += l2.cpu().data

                pred1 = torch.max(out1, 1)[1]
                pred2 = torch.max(out2, 1)[1]

                ValAcc[0] += (pred1 == distance_label).sum()
                ValBatch[0] += len(pred1)

                ValAcc[1] += (pred2 == event_label).sum()
                ValBatch[1] += len(pred2)

                label_lst[0].extend(distance_label)
                pred_lst[0].extend(pred1)
                label_lst[1].extend(event_label)
                pred_lst[1].extend(pred2)

            # print('Model Predicting Time：{}'.format(datetime.datetime.now() - t_predict))
            # return

            acc1 = float(ValAcc[0]) / ValBatch[0]
            acc2 = float(ValAcc[1]) / ValBatch[1]
            testLossLine[0].append(
                testloss[0] / len(data_loader['val'].dataset))
            testLossLine[1].append(testloss[1] / len(data_loader['val'].dataset))

            testAccLine[0] = np.append(testAccLine[0], acc1)
            testAccLine[1] = np.append(testAccLine[1], acc2)
            print("{}\nepoch:{}  Validation Accuracy: distance:{}  event:{}".format("*" * 50, epoch, acc1, acc2))

            label_lst[0] = [int(x) for x in label_lst[0]]
            pred_lst[0] = [int(x) for x in pred_lst[0]]
            label_lst[1] = [int(x) for x in label_lst[1]]
            pred_lst[1] = [int(x) for x in pred_lst[1]]

            con_mat = [confusion_matrix(label_lst[0], pred_lst[0]), confusion_matrix(label_lst[1], pred_lst[1])]

            for taskindex, taskname in enumerate(['Task 1：distance', 'Task 2：event']):
                print('{}'.format(taskname))
                print(con_mat[taskindex])
                print('Accuracy：{}'.format(accuracy_score(label_lst[taskindex], pred_lst[taskindex])))
                print(f1_score(label_lst[taskindex], pred_lst[taskindex], average=None))
                print('F1_Score：{}'.format(f1_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('Precision：{}'.format(
                    precision_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('Recall：{}'.format(recall_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))

            if save_output:
                if not is_test:
                    np.save(save_dir + '/testAccLine', testAccLine)
                    np.save(save_dir + '/testLossLine', testLossLine)

                if acc1 >= 0.98:
                    torch.save(model.state_dict(), os.path.join(save_dir,
                                                                "{}__{:.5f}_{}.pth".format(
                                                                    datetime.datetime.now().strftime(
                                                                        "%Y_%m_%d__%H_%M_%S"),
                                                                    acc1, epoch)))

                    np.save(save_dir + '/confusion matrix distance {:.5f} {}.npy'.format(acc1, epoch), con_mat[0])
                    np.save(save_dir + '/confusion matrix event {:.5f} {}.npy'.format(acc2, epoch), con_mat[1])

            if is_test:
                return

        if epoch < epoch_num - 1:

            model.train(True)

            for batch_cnt, data in enumerate(data_loader['train']):

                bacthX, distance_label, event_label = data

                if use_gpu:
                    bacthX = Variable(bacthX.cuda())
                    distance_label = Variable(distance_label.cuda())
                    event_label = Variable(event_label.cuda())
                else:
                    bacthX = Variable(bacthX)
                    distance_label = Variable(distance_label)
                    event_label = Variable(event_label)

                out1, out2 = model(bacthX)

                loss = [criterion(out1, distance_label.long()), criterion(out2, event_label.long())]

                '''
                Note: The total loss is simply the sum of the two task-specific loss,  while other
                 dynamic adaptive dynamic adjustment methods can be used.
                '''
                loss_balance = loss[0] + loss[1]

                pred1 = torch.max(out1, 1)[1]
                pred2 = torch.max(out2, 1)[1]

                optimizer.zero_grad()
                loss_balance.backward()
                optimizer.step()

                acc_sum100[0] += float(torch.sum((pred1 == distance_label)).data) / len(pred1)
                acc_sum100[1] += float(torch.sum((pred2 == event_label)).data) / len(pred2)

                loss_sum100[0] += loss[0].cpu().data.item() / len(pred1)
                loss_sum100[1] += loss[1].cpu().data.item() / len(pred2)

                if (batch_cnt + 1) % 100 == 0:
                    acc_sum100[0] = acc_sum100[0] / 100
                    loss_sum100[0] = loss_sum100[0] / 100
                    acc_sum100[1] = acc_sum100[1] / 100
                    loss_sum100[1] = loss_sum100[1] / 100

                    print("epoch-iteration:{}-{}, loss:{}, accuracy:{}".format(epoch + 1, (batch_cnt + 1), loss_sum100,
                                                                               acc_sum100))
                    print("time:{}".format(datetime.datetime.now() - start_time))

                    LossLine = [np.append(LossLine[0], loss_sum100[0]), np.append(LossLine[1], loss_sum100[1])]
                    AccLine = [np.append(AccLine[0], acc_sum100[0]), np.append(AccLine[1], acc_sum100[1])]
                    if save_output:
                        np.save(save_dir + '/trainLossLine', LossLine)
                        np.save(save_dir + '/trainAccLine', AccLine)
                    acc_sum100 = [0, 0]
                    loss_sum100 = [0, 0]

            print("Epoch {} finished！".format(epoch + 1))

            acc_sum100 = [0, 0]
            loss_sum100 = [0, 0]


def trainer_single_task(model, epoch_num, start_epoch, optimizer, criterion, data_loader, save_dir, task='event',
                        use_gpu=True, save_output=False, is_test=False):
    start_time = datetime.datetime.now()

    def adjust_lr(optimizer):
        for param in optimizer.param_groups:
            param["lr"] = param["lr"] / 1.5
        return optimizer

    LossLine = [[], []]
    AccLine = [[], []]
    testAccLine = [[], []]
    testLossLine = [[], []]
    acc_sum100 = [0, 0]
    loss_sum100 = [0, 0]

    for epoch in range(start_epoch, epoch_num):

        # Validation for every epoch
        if epoch % 5 == 0:

            optimizer = adjust_lr(optimizer)

            model.train(False)
            model.eval()
            ValAcc = [0, 0]
            ValBatch = [0, 0]
            testloss = [0, 0]

            pred_lst = [[], []]
            label_lst = [[], []]

            # t_predict = datetime.datetime.now()
            for val_cnt, val_data in enumerate(data_loader["val"]):

                valX, distance_label, event_label = val_data

                if task == 'distance':
                    label_single = distance_label
                elif task == 'event':
                    label_single = event_label
                else:
                    raise ValueError

                if use_gpu:
                    valX = Variable(valX.cuda())
                    label_single = Variable(label_single.cuda())
                    # event_label = Variable(event_label.cuda())
                else:
                    valX = Variable(valX)
                    label_single = Variable(label_single)
                    # event_label = Variable(event_label)

                out1 = model(valX)

                l1 = criterion(out1, label_single.long())
                # l2 = criterion(out2, event_label.long())

                testloss[0] += l1.cpu().data
                # testloss[1] += l2.cpu().data

                pred1 = torch.max(out1, 1)[1]
                # pred2 = torch.max(out2, 1)[1]

                ValAcc[0] += (pred1 == label_single).sum()
                ValBatch[0] += len(pred1)

                # ValAcc[1] += (pred2 == event_label).sum()
                # ValBatch[1] += len(pred2)

                label_lst[0].extend(label_single)
                pred_lst[0].extend(pred1)
                # label_lst[1].extend(event_label)
                # pred_lst[1].extend(pred2)

            # print('Model Predicting Time：{}'.format(datetime.datetime.now() - t_predict))
            # return

            acc1 = float(ValAcc[0]) / ValBatch[0]
            # acc2 = float(ValAcc[1]) / ValBatch[1]
            testLossLine[0].append(
                testloss[0] / len(data_loader['val'].dataset))
            testLossLine[1].append(
                testloss[0] / len(data_loader['val'].dataset))

            testAccLine[0] = np.append(testAccLine[0], acc1)
            testAccLine[1] = np.append(testAccLine[1], acc1)
            print("{}\nepoch:{}  Validation Accuracy: ".format("*" * 50, epoch, acc1))

            label_lst[0] = [int(x) for x in label_lst[0]]
            pred_lst[0] = [int(x) for x in pred_lst[0]]
            # label_lst[1] = [int(x) for x in label_lst[1]]
            # pred_lst[1] = [int(x) for x in pred_lst[1]]

            con_mat = [confusion_matrix(label_lst[0], pred_lst[0]), confusion_matrix(label_lst[1], pred_lst[1])]

            for taskindex, taskname in enumerate(['Task 1：distance']):
                print('{}'.format(taskname))
                print(con_mat[taskindex])
                print('Accuracy：{}'.format(accuracy_score(label_lst[taskindex], pred_lst[taskindex])))
                print(f1_score(label_lst[taskindex], pred_lst[taskindex], average=None))
                print('F1_Score：{}'.format(f1_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('Precision：{}'.format(
                    precision_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('Recall：{}'.format(recall_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))

            if save_output:
                if not is_test:
                    np.save(save_dir + '/testAccLine', testAccLine)
                    np.save(save_dir + '/testLossLine', testLossLine)

                if acc1 >= 0.98:
                    torch.save(model.state_dict(), os.path.join(save_dir,
                                                                "{}__{:.5f}_{}.pth".format(
                                                                    datetime.datetime.now().strftime(
                                                                        "%Y_%m_%d__%H_%M_%S"),
                                                                    acc1, epoch)))

                    np.save(save_dir + '/confusion matrix distance {:.5f} {}.npy'.format(acc1, epoch), con_mat[0])
                    # np.save(save_dir + '/confusion matrix event {:.5f} {}.npy'.format(acc2, epoch), con_mat[1])

            if is_test:
                return

        if epoch < epoch_num - 1:

            model.train(True)

            for batch_cnt, data in enumerate(data_loader['train']):

                bacthX, distance_label, event_label = data

                if task == 'distance':
                    label_single = distance_label
                elif task == 'event':
                    label_single = event_label
                else:
                    raise ValueError

                if use_gpu:
                    bacthX = Variable(bacthX.cuda())
                    label_single = Variable(label_single.cuda())
                    # event_label = Variable(event_label.cuda())
                else:
                    bacthX = Variable(bacthX)
                    label_single = Variable(label_single)
                    # event_label = Variable(event_label)

                out1 = model(bacthX)

                loss = criterion(out1, label_single.long())

                loss_balance = loss

                pred1 = torch.max(out1, 1)[1]
                # pred2 = torch.max(out2, 1)[1]

                optimizer.zero_grad()
                loss_balance.backward()
                optimizer.step()

                acc_sum100[0] += float(torch.sum((pred1 == label_single)).data) / len(pred1)
                # acc_sum100[1] += float(torch.sum((pred2 == event_label)).data) / len(pred2)

                loss_sum100[0] += loss.cpu().data.item() / len(pred1)
                # loss_sum100[1] += loss[1].cpu().data.item() / len(pred2)

                if (batch_cnt + 1) % 100 == 0:
                    acc_sum100[0] = acc_sum100[0] / 100
                    loss_sum100[0] = loss_sum100[0] / 100
                    # acc_sum100[1] = acc_sum100[1] / 100
                    # loss_sum100[1] = loss_sum100[1] / 100

                    print("epoch-iteration:{}-{}, loss:{}, accuracy:{}".format(epoch + 1, (batch_cnt + 1), loss_sum100,
                                                                               acc_sum100))
                    print("time:{}".format(datetime.datetime.now() - start_time))

                    LossLine = [np.append(LossLine[0], loss_sum100[0])]
                    AccLine = [np.append(AccLine[0], acc_sum100[0])]
                    if save_output:
                        np.save(save_dir + '/trainLossLine', LossLine)
                        np.save(save_dir + '/trainAccLine', AccLine)
                    acc_sum100 = [0, 0]
                    loss_sum100 = [0, 0]

            print("Epoch {} finished！".format(epoch + 1))

            acc_sum100 = [0, 0]
            loss_sum100 = [0, 0]


def trainer_multiClassifier(model, epoch_num, start_epoch, optimizer, criterion, data_loader, save_dir,
                            use_gpu=True, save_output=False, is_test=False):
    # Establish mapping from multi-categories to categories of two tasks
    hash_list = [[i % 16, i // 16] for i in range(32)]
    start_time = datetime.datetime.now()

    if is_test:
        print('Start Test：{}'.format(start_time))
    else:
        print('Start Training：{}'.format(start_time))

    def adjust_lr(optimizer):
        for param in optimizer.param_groups:
            param["lr"] = param["lr"] / 1.5
        return optimizer

    LossLine = [[], []]
    AccLine = [[], []]
    testAccLine = [[], []]
    testLossLine = [[], []]
    acc_sum100 = [0, 0]
    loss_sum100 = [0, 0]

    for epoch in range(start_epoch, epoch_num):

        if epoch % 5 == 0:

            if epoch != 0:
                optimizer = adjust_lr(optimizer)

            model.train(False)
            model.eval()
            ValAcc = [0, 0]
            ValBatch = [0, 0]
            testloss = [0, 0]
            pred_lst = [[], []]
            label_lst = [[], []]

            # t_predict = datetime.datetime.now()
            for val_cnt, val_data in enumerate(data_loader["val"]):

                valX, mix_label = val_data

                if use_gpu:
                    valX = Variable(valX.cuda())
                    mix_label = Variable(mix_label.cuda())

                else:
                    valX = Variable(valX)
                    mix_label = Variable(mix_label)

                out1 = model(valX)
                out1_pred = torch.max(out1, 1)[1]
                l1 = criterion(out1, mix_label.long())

                testloss[0] += l1.cpu().data
                testloss[1] += l1.cpu().data

                # - Transform multi-categories to categories of two tasks
                pred1 = torch.zeros(out1_pred.shape[0])
                pred2 = torch.zeros(out1_pred.shape[0])
                for switch_i in range(out1_pred.shape[0]):
                    pred1[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][0]
                    pred2[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][1]

                distance_label = torch.zeros(mix_label.shape[0])
                event_label = torch.zeros(mix_label.shape[0])
                for switch_i in range(mix_label.shape[0]):
                    distance_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][0]
                    event_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][1]

                ValAcc[0] += (pred1 == distance_label).sum()
                ValBatch[0] += len(pred1)

                ValAcc[1] += (pred2 == event_label).sum()
                ValBatch[1] += len(pred2)

                label_lst[0].extend(distance_label)
                pred_lst[0].extend(pred1)
                label_lst[1].extend(event_label)
                pred_lst[1].extend(pred2)

            # print('Predicting time：{}'.format(datetime.datetime.now() - t_predict))
            # return

            acc1 = float(ValAcc[0]) / ValBatch[0]
            acc2 = float(ValAcc[1]) / ValBatch[1]

            testLossLine[0].append(
                testloss[0] / len(data_loader['val'].dataset))
            testLossLine[1].append(testloss[1] / len(data_loader['val'].dataset))

            testAccLine[0] = np.append(testAccLine[0], acc1)
            testAccLine[1] = np.append(testAccLine[1], acc2)
            print("{}\nepoch:{}  Accuracy: distance:{}  event:{}".format("*" * 50, epoch + 1, acc1, acc2))

            label_lst[0] = [int(x) for x in label_lst[0]]
            pred_lst[0] = [int(x) for x in pred_lst[0]]
            label_lst[1] = [int(x) for x in label_lst[1]]
            pred_lst[1] = [int(x) for x in pred_lst[1]]

            con_mat = [confusion_matrix(label_lst[0], pred_lst[0]), confusion_matrix(label_lst[1], pred_lst[1])]

            for taskindex, taskname in enumerate(['Task 1：distance', 'Task 2：event']):
                print('{}'.format(taskname))
                print(con_mat[taskindex])
                print('Accuracy：{}'.format(accuracy_score(label_lst[taskindex], pred_lst[taskindex])))
                print(f1_score(label_lst[taskindex], pred_lst[taskindex], average=None))
                print('F1_Score：{}'.format(f1_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('Precision：{}'.format(
                    precision_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('Recall：{}'.format(recall_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))

            if save_output:

                if not is_test:
                    np.save(save_dir + '/testAccLine', testAccLine)
                    np.save(save_dir + '/testLossLine', testLossLine)

                if acc1 >= 0.95:
                    torch.save(model.state_dict(), os.path.join(save_dir,
                                                                "{}__{:.5f}_{}.pth".format(
                                                                    datetime.datetime.now().strftime(
                                                                        "%Y_%m_%d__%H_%M_%S"),
                                                                    acc1, epoch)))
                    np.save(save_dir + '/confusion matrix {} {}.npy'.format(acc1, epoch), con_mat[0])  #
                    np.save(save_dir + '/confusion matrix {} {}.npy'.format(acc2, epoch), con_mat[1])  #
            if is_test:
                return

        if epoch < epoch_num - 1:

            model.train(True)

            for batch_cnt, data in enumerate(data_loader['train']):

                bacthX, mix_label = data

                if use_gpu:
                    bacthX = Variable(bacthX.cuda())
                    mix_label = Variable(mix_label.cuda())
                    # event_label = Variable(event_label.cuda())
                else:
                    bacthX = Variable(bacthX)
                    mix_label = Variable(mix_label)
                    # event_label = Variable(event_label)

                out1 = model(bacthX)

                loss = [criterion(out1, mix_label.long())]
                loss_balance = loss[0]

                optimizer.zero_grad()
                loss_balance.backward()
                optimizer.step()

                out1_pred = torch.max(out1, 1)[1]

                pred1 = torch.zeros(out1_pred.shape[0])
                pred2 = torch.zeros(out1_pred.shape[0])
                for switch_i in range(out1_pred.shape[0]):
                    pred1[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][0]
                    pred2[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][1]

                distance_label = torch.zeros(mix_label.shape[0])
                event_label = torch.zeros(mix_label.shape[0])
                for switch_i in range(mix_label.shape[0]):
                    distance_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][0]
                    event_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][1]

                acc_sum100[0] += float(torch.sum((pred1 == distance_label)).data) / len(pred1)
                acc_sum100[1] += float(torch.sum((pred2 == event_label)).data) / len(pred2)

                loss_sum100[0] += loss[0].cpu().data.item() / len(out1_pred)
                loss_sum100[1] += loss[0].cpu().data.item() / len(out1_pred)

                if (batch_cnt + 1) % 100 == 0:
                    acc_sum100[0] = acc_sum100[0] / 100
                    loss_sum100[0] = loss_sum100[0] / 100
                    acc_sum100[1] = acc_sum100[1] / 100
                    loss_sum100[1] = loss_sum100[1] / 100

                    print("epoch-iteration:{}-{}, loss:{}, accuracy:{}".format(epoch + 1, (batch_cnt + 1), loss_sum100,
                                                                               acc_sum100))
                    print("time:{}".format(datetime.datetime.now() - start_time))

                    LossLine = [np.append(LossLine[0], loss_sum100[0]), np.append(LossLine[0], loss_sum100[0])]
                    AccLine = [np.append(AccLine[0], acc_sum100[0]), np.append(AccLine[1], acc_sum100[1])]
                    if save_output:
                        np.save(save_dir + '/trainLossLine', LossLine)
                        np.save(save_dir + '/trainAccLine', AccLine)
                    acc_sum100 = [0, 0]
                    loss_sum100 = [0, 0]

            print("Epoch {} finished！".format(epoch + 1))
            acc_sum100 = [0, 0]
            loss_sum100 = [0, 0]
