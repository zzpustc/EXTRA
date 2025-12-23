from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from curves import Conv2d, Linear, BatchNorm2d


class Conv_Layer(nn.Module):
    def __init__(self, channel, fix_points, pred=False):
        super(Conv_Layer, self).__init__()
        if not pred:
            self.conv1 = Conv2d(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=3,
                padding=1,
                fix_points = fix_points
            )
            self.bn = BatchNorm2d(num_features=channel[1], fix_points = fix_points)
            self.relu = nn.ReLU(inplace=True)
        else:

            self.conv1 = Conv2d(
                in_channels=channel[0],
                out_channels=channel[0],
                kernel_size=3,
                padding=1,
                fix_points = fix_points
            )
            self.conv2 = Conv2d(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=1,
                padding=0,
                fix_points = fix_points
            )
        self.pred = pred

    def forward(self, x, coeffs_t):
        if not self.pred:
            x = self.conv1(x, coeffs_t)
            x = self.bn(x, coeffs_t)
            out = self.relu(x)
            return out
        else:
            x = self.conv1(x, coeffs_t)
            out = self.conv2(x, coeffs_t)
            return out

class Att_Layer(nn.Module):
    def __init__(self, channel, fix_points):
        super(Att_Layer, self).__init__()

        self.conv1 = Conv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=1,
            padding=0,
            fix_points = fix_points
        )
        self.bn1 = BatchNorm2d(channel[1], fix_points = fix_points)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(
            in_channels=channel[1],
            out_channels=channel[2],
            kernel_size=1,
            padding=0,
            fix_points = fix_points
        )
        self.bn2 = BatchNorm2d(channel[2], fix_points = fix_points)
        self.sigmoid1 = nn.Sigmoid()
        
    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.relu1(x)

        x = self.conv2(x, coeffs_t)
        x = self.bn2(x, coeffs_t)     

        out = self.sigmoid1(x)   

        return out
    
class Two_Conv_Layer(nn.Module):
    def __init__(self, channel, fix_points):
        super(Two_Conv_Layer, self).__init__()

        conv_layer = Conv_Layer

        self.conv_block1 = conv_layer(channel, fix_points)
        self.conv_block2 = conv_layer(channel, fix_points)

    def forward(self, x, coeffs_t):

        x = self.conv_block1(x, coeffs_t)
        out = self.conv_block2(x, coeffs_t)

        return out


class _SegNet(nn.Module):
    """SegNet MTAN"""

    def __init__(self, fix_points):
        super(_SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        conv_layer = Conv_Layer
        att_layer = Att_Layer
        two_conv_layer = Two_Conv_Layer

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([conv_layer([3, filter[0]], fix_points)])
        self.decoder_block = nn.ModuleList([conv_layer([filter[0], filter[0]], fix_points)])
        for i in range(4):
            self.encoder_block.append(conv_layer([filter[i], filter[i + 1]], fix_points))
            self.decoder_block.append(conv_layer([filter[i + 1], filter[i]], fix_points))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([conv_layer([filter[0], filter[0]], fix_points)])
        self.conv_block_dec = nn.ModuleList([conv_layer([filter[0], filter[0]], fix_points)])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    conv_layer([filter[i + 1], filter[i + 1]], fix_points)
                )
                self.conv_block_dec.append(conv_layer([filter[i], filter[i]], fix_points))
            else:
                self.conv_block_enc.append(
                    two_conv_layer([filter[i + 1], filter[i + 1]], fix_points)
                )
                self.conv_block_dec.append(
                    two_conv_layer([filter[i], filter[i]], fix_points)
                )

        # define task attention layers
        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([att_layer([filter[0], filter[0], filter[0]], fix_points)])]
        )
        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([att_layer([2 * filter[0], filter[0], filter[0]], fix_points)])]
        )
        self.encoder_block_att = nn.ModuleList(
            [conv_layer([filter[0], filter[1]], fix_points)]
        )
        self.decoder_block_att = nn.ModuleList(
            [conv_layer([filter[0], filter[0]], fix_points)]
        )

        for j in range(3):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([att_layer([filter[0], filter[0], filter[0]], fix_points)])
                )
                self.decoder_att.append(
                    nn.ModuleList(
                        [att_layer([2 * filter[0], filter[0], filter[0]], fix_points)]
                    )
                )
            for i in range(4):
                self.encoder_att[j].append(
                    att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]], fix_points)
                )
                self.decoder_att[j].append(
                    att_layer([filter[i + 1] + filter[i], filter[i], filter[i]], fix_points)
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(
                    conv_layer([filter[i + 1], filter[i + 2]], fix_points)
                )
                self.decoder_block_att.append(
                    conv_layer([filter[i + 1], filter[i]], fix_points)
                )
            else:
                self.encoder_block_att.append(
                    conv_layer([filter[i + 1], filter[i + 1]], fix_points)
                )
                self.decoder_block_att.append(
                    conv_layer([filter[i + 1], filter[i + 1]], fix_points)
                )

        self.pred_task1 = conv_layer([filter[0], self.class_nb], fix_points, pred=True)
        self.pred_task2 = conv_layer([filter[0], 1], fix_points, pred=True)
        self.pred_task3 = conv_layer([filter[0], 3], fix_points, pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)


    def shared_modules(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            # self.encoder_att, self.decoder_att,
            self.encoder_block_att,
            self.decoder_block_att,
            self.down_sampling,
            self.up_sampling,
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()


    def forward(self, x, coeffs_t):  
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x, coeffs_t)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0], coeffs_t)
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1], coeffs_t)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0], coeffs_t)
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i], coeffs_t)
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0], coeffs_t)
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i], coeffs_t)
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0], coeffs_t)

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0], coeffs_t)
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1], coeffs_t
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1), coeffs_t
                    )
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1], coeffs_t
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_encoder[i][-1][-1],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0], coeffs_t
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1), coeffs_t
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_decoder[i][j - 1][2],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0], coeffs_t
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1), coeffs_t
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1], coeffs_t), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1], coeffs_t)
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1], coeffs_t)
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return (
            [t1_pred, t2_pred, t3_pred],
            (
                atten_decoder[0][-1][-1],
                atten_decoder[1][-1][-1],
                atten_decoder[2][-1][-1],
            ),
        )



class SegNetMtan_Curve(nn.Module):
    def __init__(self, fix_points):
        super().__init__()
        self.segnet = _SegNet(fix_points)

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" not in n)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" in n)

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return []

    def forward(self, x, coeffs_t, return_representation=False):
        if return_representation:
            return self.segnet(x, coeffs_t)
        else:
            pred, rep = self.segnet(x, coeffs_t)
            return pred

