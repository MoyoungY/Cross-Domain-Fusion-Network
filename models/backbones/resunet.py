import torch
import torch.nn as nn
import math


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        if stride == 1:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_dim),
            )
        elif stride == 2:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(output_dim),
            )
        self.rl = nn.ReLU()

    def forward(self, x):
        return self.rl(self.conv_block(x) + self.conv_skip(x))


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()


        self.upsample = [nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )]
        self.upsample.append(nn.BatchNorm2d(output_dim))
        self.upsample.append(nn.ReLU())
        self.upsample = nn.Sequential(*self.upsample)

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, in_channel, out_channel, filters=[8, 16, 32, 64]):
        super(ResUnet, self).__init__()

        self.input_layer = ResidualConv(in_channel, filters[0], 1, 1)

        self.residual_down_21 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_down_31 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_down_41 = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_32 = Upsample(filters[3], filters[2], 2, 2)
        self.residual_conv32 = ResidualConv(2*filters[2], filters[2], 1, 1)

        self.upsample_22 = Upsample(filters[2], filters[1], 2, 2)
        self.residual_conv22 = ResidualConv(2*filters[1], filters[1], 1, 1)
        self.upsample_23 = Upsample(filters[2], filters[1], 2, 2)
        self.residual_conv23 = ResidualConv(3*filters[1], filters[1], 1, 1)

        self.upsample_12 = Upsample(filters[1], filters[0], 2, 2)
        self.residual_conv12 = ResidualConv(2*filters[0], filters[0], 1, 1)
        self.upsample_13 = Upsample(filters[1], filters[0], 2, 2)
        self.residual_conv13 = ResidualConv(3*filters[0], filters[0], 1, 1)
        self.upsample_14 = Upsample(filters[1], filters[0], 2, 2)
        self.residual_conv14 = ResidualConv(4*filters[0], filters[0], 1, 1)

        self.output_layer1 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )
        self.output_layer3 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )
        self.output_layer4 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )

    def forward(self, x):
        # Encode
        x1_1 = self.input_layer(x)
        x2_1 = self.residual_down_21(x1_1)
        x3_1 = self.residual_down_31(x2_1)
        x4_1 = self.residual_down_41(x3_1)
        # Decode
        x3_2 = self.upsample_32(x4_1)
        x3_2 = torch.cat([x3_1, x3_2], dim=1)
        x3_2 = self.residual_conv32(x3_2)

        x2_2 = self.upsample_22(x3_1)
        x2_2 = torch.cat([x2_1, x2_2], dim=1)
        x2_2 = self.residual_conv22(x2_2)
        x2_3 = self.upsample_23(x3_2)
        x2_3 = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x2_3 = self.residual_conv23(x2_3)

        x1_2 = self.upsample_12(x2_1)
        x1_2 = torch.cat([x1_1, x1_2], dim=1)
        x1_2 = self.residual_conv12(x1_2)
        x1_3 = self.upsample_13(x2_2)
        x1_3 = torch.cat([x1_1, x1_2, x1_3], dim=1)
        x1_3 = self.residual_conv13(x1_3)
        x1_4 = self.upsample_14(x2_3)
        x1_4 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1_4 = self.residual_conv14(x1_4)

        # output1_1 = self.output_layer1(x1_1)
        # output1_2 = self.output_layer2(x1_2)
        # output1_3 = self.output_layer3(x1_3)
        output1_4 = self.output_layer4(x1_4)

        return output1_4

class ResUnet_DS(nn.Module):
    def __init__(self, in_channel, out_channel, filters=[8, 16, 32, 64]):
        super(ResUnet_DS, self).__init__()

        self.input_layer = ResidualConv(in_channel, filters[0], 1, 1)

        self.residual_down_21 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_down_31 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_down_41 = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_32 = Upsample(filters[3], filters[2], 2, 2)
        self.residual_conv32 = ResidualConv(2*filters[2], filters[2], 1, 1)

        self.upsample_22 = Upsample(filters[2], filters[1], 2, 2)
        self.residual_conv22 = ResidualConv(2*filters[1], filters[1], 1, 1)
        self.upsample_23 = Upsample(filters[2], filters[1], 2, 2)
        self.residual_conv23 = ResidualConv(3*filters[1], filters[1], 1, 1)

        self.upsample_12 = Upsample(filters[1], filters[0], 2, 2)
        self.residual_conv12 = ResidualConv(2*filters[0], filters[0], 1, 1)
        self.upsample_13 = Upsample(filters[1], filters[0], 2, 2)
        self.residual_conv13 = ResidualConv(3*filters[0], filters[0], 1, 1)
        self.upsample_14 = Upsample(filters[1], filters[0], 2, 2)
        self.residual_conv14 = ResidualConv(4*filters[0], filters[0], 1, 1)

        self.output_layer1 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )
        self.output_layer3 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )
        self.output_layer4 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Hardtanh(-math.pi, math.pi)
        )

    def forward(self, x):
        # Encode
        x1_1 = self.input_layer(x)
        x2_1 = self.residual_down_21(x1_1)
        x3_1 = self.residual_down_31(x2_1)
        x4_1 = self.residual_down_41(x3_1)
        # Decode
        x3_2 = self.upsample_32(x4_1)
        x3_2 = torch.cat([x3_1, x3_2], dim=1)
        x3_2 = self.residual_conv32(x3_2)

        x2_2 = self.upsample_22(x3_1)
        x2_2 = torch.cat([x2_1, x2_2], dim=1)
        x2_2 = self.residual_conv22(x2_2)
        x2_3 = self.upsample_23(x3_2)
        x2_3 = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x2_3 = self.residual_conv23(x2_3)

        x1_2 = self.upsample_12(x2_1)
        x1_2 = torch.cat([x1_1, x1_2], dim=1)
        x1_2 = self.residual_conv12(x1_2)
        x1_3 = self.upsample_13(x2_2)
        x1_3 = torch.cat([x1_1, x1_2, x1_3], dim=1)
        x1_3 = self.residual_conv13(x1_3)
        x1_4 = self.upsample_14(x2_3)
        x1_4 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1_4 = self.residual_conv14(x1_4)

        output1_1 = self.output_layer1(x1_1)
        output1_2 = self.output_layer2(x1_2)
        output1_3 = self.output_layer3(x1_3)
        output1_4 = self.output_layer4(x1_4)

        return output1_1, output1_2, output1_3, output1_4
