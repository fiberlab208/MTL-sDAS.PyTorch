import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(dilation, dilation), groups=groups, bias=False,
                     dilation=(dilation, dilation))


def att_generator(in_channel, intermediate_channel, out_channel):
    """The attention mask generator"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=(1, 1)),  # 只用于降维
        nn.BatchNorm2d(intermediate_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.Sigmoid())


class Single_Task_Net(nn.Module):
    """
    Implementation of the Single Task network (Model B in the paper)
    """

    def __init__(self, task='distance'):
        super(Single_Task_Net, self).__init__()
        self.res_num = 8
        self.first_ch = 16

        self.ch = [self.first_ch, self.first_ch]
        for i in range(self.res_num // 2 - 1):
            self.ch.append(self.first_ch * (2 ** (i + 1)))  # The number of channels doubles every two blocks

        self.tasks = [task]  # only one task
        self.task_cate_num = [16, 2]  # Number of categories for two tasks

        """Define The shared backbone network"""
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.ch[0], kernel_size=(7, 7), stride=(3, 3), padding=(2, 2), bias=False),
            nn.BatchNorm2d(self.ch[0]),
            nn.ReLU()
        )

        # Resblocks
        self.resblock1 = ResBlock(inchannel=self.ch[0], outchannel=self.ch[1], stride=1)
        self.resblock2 = ResBlock(inchannel=self.ch[1], outchannel=self.ch[1], stride=1)
        self.resblock3 = ResBlock(inchannel=self.ch[1], outchannel=self.ch[2], stride=2)
        self.resblock4 = ResBlock(inchannel=self.ch[2], outchannel=self.ch[2], stride=1)
        self.resblock5 = ResBlock(inchannel=self.ch[2], outchannel=self.ch[3], stride=2)
        self.resblock6 = ResBlock(inchannel=self.ch[3], outchannel=self.ch[3], stride=1)
        self.resblock7 = ResBlock(inchannel=self.ch[3], outchannel=self.ch[4], stride=2)
        self.resblock8 = ResBlock(inchannel=self.ch[4], outchannel=self.ch[4], stride=1)

        """Define the task-specific networks"""
        # Attention mask generator
        self.att_mask_generator1 = nn.ModuleList(
            [att_generator(self.ch[1], self.ch[1] // 2, self.ch[1]) for _ in self.tasks])
        self.att_mask_generato2 = nn.ModuleList(
            [att_generator(2 * self.ch[2], self.ch[2] // 2, self.ch[2]) for _ in self.tasks])
        self.att_mask_generator3 = nn.ModuleList(
            [att_generator(2 * self.ch[3], self.ch[3] // 2, self.ch[3]) for _ in self.tasks])
        self.att_mask_generator4 = nn.ModuleList(
            [att_generator(2 * self.ch[4], self.ch[4] // 2, self.ch[4]) for _ in self.tasks])

        # Define the output layer
        self.output_layer1 = nn.ModuleList(
            [nn.Sequential(conv3x3(self.ch[1], self.ch[2], stride=1), nn.BatchNorm2d(self.ch[2]), nn.ReLU(inplace=True))
             for _ in self.tasks])

        self.output_layer2 = nn.ModuleList(
            [nn.Sequential(conv3x3(self.ch[2], self.ch[3], stride=1), nn.BatchNorm2d(self.ch[3]),
                           nn.ReLU(inplace=True)) for _ in
             self.tasks])

        self.output_layer3 = nn.ModuleList(
            [nn.Sequential(conv3x3(self.ch[3], self.ch[4], stride=1), nn.BatchNorm2d(self.ch[4]),
                           nn.ReLU(inplace=True)) for _ in
             self.tasks])

        # Maxpool layer for the output layer
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #  The output layer

        if self.tasks[0] == 'distance':
            self.task1pool = nn.AdaptiveAvgPool2d((1, 1))
            self.task1pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // self.task_cate_num[0],
                                            stride=self.ch[-1] // self.task_cate_num[0])
        elif self.tasks[0] == 'event':
            self.task2pool = nn.AdaptiveAvgPool2d((1, 1))
            self.task2pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // self.task_cate_num[1],
                                            stride=self.ch[-1] // self.task_cate_num[1])
        else:
            raise ValueError

    def forward(self, x):
        x = self.conv1(x)

        shared_blocks = []  # A list for all the shared Resblocks

        shared_blocks.append(self.resblock1(x))
        shared_blocks.append(self.resblock2(shared_blocks[-1]))
        shared_blocks.append(self.resblock3(shared_blocks[-1]))
        shared_blocks.append(self.resblock4(shared_blocks[-1]))
        shared_blocks.append(self.resblock5(shared_blocks[-1]))
        shared_blocks.append(self.resblock6(shared_blocks[-1]))
        shared_blocks.append(self.resblock7(shared_blocks[-1]))
        shared_blocks.append(self.resblock8(shared_blocks[-1]))

        a_1_mask = [att_i(shared_blocks[0]) for att_i in
                    self.att_mask_generator1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * shared_blocks[1] for a_1_mask_i in
               a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(encoder_block(a_1_i)) for a_1_i, encoder_block in
               # Obtain the output of the attention module
               zip(a_1, self.output_layer1)]

        a_2_mask = [att_i(torch.cat((shared_blocks[2], a_1_i), dim=1)) for a_1_i, att_i in
                    zip(a_1, self.att_mask_generato2)]
        a_2 = [a_2_mask_i * shared_blocks[3] for a_2_mask_i in a_2_mask]
        a_2 = [self.down_sampling(encoder_block(a_2_i)) for a_2_i, encoder_block in
               zip(a_2, self.output_layer2)]

        a_3_mask = [att_i(torch.cat((shared_blocks[4], a_2_i), dim=1)) for a_2_i, att_i in
                    zip(a_2, self.att_mask_generator3)]
        a_3 = [a_3_mask_i * shared_blocks[5] for a_3_mask_i in a_3_mask]
        a_3 = [self.down_sampling(encoder_block(a_3_i)) for a_3_i, encoder_block in
               zip(a_3, self.output_layer3)]

        a_4_mask = [att_i(torch.cat((shared_blocks[6], a_3_i), dim=1)) for a_3_i, att_i in
                    zip(a_3, self.att_mask_generator4)]
        a_4 = [a_4_mask_i * shared_blocks[7] for a_4_mask_i in a_4_mask]

        if self.tasks[0] == 'distance':
            pred1 = self.task1pool(a_4[0]).squeeze(2).squeeze(2)
            pred1 = self.task1pool1d(pred1.unsqueeze(1)).squeeze(1)
            pred1 = F.log_softmax(pred1, dim=1)
        elif self.tasks[0] == 'event':
            pred1 = self.task2pool(a_4[0]).squeeze(2).squeeze(2)
            pred1 = self.task2pool1d(pred1.unsqueeze(1)).squeeze(1)
            pred1 = F.log_softmax(pred1, dim=1)
        else:
            raise ValueError
        return pred1
