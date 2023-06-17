import warnings
import torch.utils.data
import torch
import torch.nn.functional as F
from torchvision.models import InceptionOutputs
import torch.nn as nn
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionAux, InceptionE


class BasicConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Multi_Classifier(nn.Module):
    """
    The multi classifier with standard InceptionV3 structure in Pytorch
    """

    def __init__(
            self,
            num_classes: int = 32,
            aux_logits: bool = False,
            transform_input: bool = False,
            inception_blocks=None,
            init_weights=True
    ) -> None:
        super(Multi_Classifier, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(1, 32, kernel_size=3, stride=2)  # 将这里改成1
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        # x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

# def main(is_test=False, pth_file='',random_state=1,fold_index=2):
#     note = 'InceptionV3 is_test={}'.format(is_test)  # 直接跑
#     save_dir = r'\\121.48.161.226\kb208datapool\LabFiles\users\wyf\保存结果\{}{}/'.format(
#         datetime.datetime.now().strftime("%m月%d日%H_%M_%S"), note)
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     if not os.path.exists(save_dir + 'src'):
#         os.makedirs(save_dir + 'src')
#
#     py_file = glob.glob(r'./*.py')
#     for filename in py_file:
#         copyfile(filename, save_dir + 'src/' + os.path.basename(filename))
#
#     log1 = Logger('训练过程.log', path=save_dir)
#     sys.stdout = log1
#
#     print(__file__)
#     # 定义一个哈希表
#
#     GPU_device = True
#     if not torch.cuda.is_available():
#         GPU_device = False
#
#     batchsize = 32
#     numworks = 0
#     model = Inception3()
#
#     if not pth_file == '':
#         model.load_state_dict(torch.load(pth_file, map_location='cpu'), strict=True)
#
#     if GPU_device == True:
#         model.cuda()
#
#     print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model),
#                                                              count_parameters(model) / 11689512))
#     print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model, grad=False),
#                                                              count_parameters(model, grad=False) / 11689512))
#
#     print('')
#
#     macs, params = get_model_complexity_info(model, (1, 100, 250), as_strings=False,
#                                              print_per_layer_stat=False, verbose=True, ost=log1)
#     print('{:<30}  {:<8}'.format('MACs: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#
#     base_lr = 0.0008
#     optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)
#     # criterion = torch.nn.NLLLoss()
#     criterion = torch.nn.CrossEntropyLoss()
#
#     if is_test:
#         dataset_dir0324 = r'E:\dataset\原始时空矩阵降采样增广0324qiaoji分训练测试/test/'
#         dataset_dirwajue = r'E:\dataset\原始时空矩阵降采样增广wajue分训练测试/test/'
#     else:
#         dataset_dir0324 = r'E:\dataset\原始时空矩阵降采样增广0324qiaoji分训练测试/train/'
#         dataset_dirwajue = r'E:\dataset\原始时空矩阵降采样增广wajue分训练测试/train/'
#
#     dataset_MTL = Dataset_mat_MTL(dataset_dir0324=dataset_dir0324, dataset_dirwajue=dataset_dirwajue, ram=True,
#                                   paper_single=True,is_test=is_test,random_state=random_state,fold_index=fold_index)
#
#     dataloader1 = {}
#     dataloader1['train'] = torch.utils.data.DataLoader(dataset_MTL.dataset['train'], batch_size=batchsize,
#                                                        shuffle=True, num_workers=numworks)
#     dataloader1['val'] = torch.utils.data.DataLoader(dataset_MTL.dataset['val'], batch_size=batchsize,
#                                                      shuffle=True, num_workers=numworks)
#
#     # 保存样本名-标签列表
#     dataset_MTL.dataset['train'].get_name_label_csv(savedir='train name label.csv', paper_single=True)
#     dataset_MTL.dataset['val'].get_name_label_csv(savedir='val name label.csv', paper_single=True)
#
#     is_train = True
#     if is_train:
#         train(model=model, data_loader=dataloader1, epoch_num=41, start_epoch=0, optimizer=optimizer,
#               criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True,is_test=is_test)
#
#     # -绘制四种曲线并保存
#
#     if not is_test:
#
#         linelist = ['trainAccLine', 'trainLossLine',
#                     'testAccLine', 'testLossLine']
#         for linename in linelist:
#             line = np.load(save_dir + linename + '.npy')
#             plt.figure()
#             plt.plot(line[0], label='distance')
#             plt.plot(line[1], label='event')
#             plt.legend()
#             plt.savefig(save_dir + linename + '.png')
#             plt.close()
#     else:
#         leibie_event = ['Striking', 'Digging']
#         leibei_distance = ['{}m'.format(i) for i in range(16)]
#         cm_list = glob.glob(save_dir + 'confusion matrix*.npy')
#         for cm in cm_list:
#             mat1 = np.load(cm)
#             if len(mat1[0]) == 16:
#                 draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibei_distance, figsize=(7, 6.8),
#                                       savepath=save_dir + '/confusion matrix distance.svg')
#             elif len(mat1[0]) == 2:
#                 draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibie_event, figsize=(2.8, 2.8),
#                                       savepath=save_dir + '/confusion matrix event.svg')
#             else:
#                 raise ValueError
#
#     # -绘制四种曲线并保存
#     log1.save()
#
#     if not is_test:
#
#         line = np.load(save_dir + 'testAccLine.npy')
#         with open('./' + 'accuracylist.txt', 'a') as f:
#             f.write('\n'+note+'\n')
#             f.write(str(np.max(line, axis=1)[0]))
#             f.write('\n')
#             f.write(str(np.max(line, axis=1)[1]))
#
#
# if __name__ == '__main__':
#     main()
