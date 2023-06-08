import torch
import torch.nn as nn


class InceptionConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.double_conv4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_Inc = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = [self.double_conv1(x), self.double_conv2(x), self.double_conv3(x), self.double_conv4(x)]
        output2 = self.conv_Inc(torch.cat(outputs, 1))
        xx = output2 + self.conv_skip(x)
        xx_o = self.relu(xx)
        return xx_o


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpSample, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")

    def forward(self, down_input):
        x = self.up_sample(down_input)
        return x


class UnetGridGatingSignal(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(UnetGridGatingSignal, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=1, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        xx = self.conv_block(x)
        return xx


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.UpSampling2D = nn.Upsample(scale_factor=2)

        self.conv = nn.Sequential(
            nn.Conv2d(n_coefficients, n_coefficients, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_coefficients)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """

        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        upsample_sigmoid_xg = self.UpSampling2D(psi)
        out = skip_connection * upsample_sigmoid_xg.expand_as(skip_connection)
        return out


class base_Unet(nn.Module):
    def __init__(self, img_ch=1, output_ch=4, filters=[32, 64, 128, 256, 512, 1024]):
        super(base_Unet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=None)
        self.Dropout = nn.Dropout(p=0.1, inplace=True)

        self.conv_1 = InceptionConv(img_ch, filters[0])
        self.conv_2 = InceptionConv(filters[0], filters[1])
        self.conv_3 = InceptionConv(filters[1], filters[2])
        self.conv_4 = InceptionConv(filters[2], filters[3])
        self.conv_5 = InceptionConv(filters[3], filters[4])
        self.conv_6 = InceptionConv(filters[4], filters[5])

        self.gating = UnetGridGatingSignal(filters[5], filters[4], 1, 0)

        self.upsample1 = UpSample(filters[5], filters[5], up_sample_mode='bilinear')
        self.Att1 = AttentionBlock(F_g=filters[5], F_l=filters[4], n_coefficients=filters[4])
        self.up_conv1 = InceptionConv(filters[5] + filters[4], filters[4])

        self.upsample2 = UpSample(filters[4], filters[4], up_sample_mode='bilinear')
        self.Att2 = AttentionBlock(F_g=filters[4], F_l=filters[3], n_coefficients=filters[3])
        self.up_conv2 = InceptionConv(filters[4] + filters[3], filters[3])

        self.upsample3 = UpSample(filters[3], filters[3], up_sample_mode='bilinear')
        self.Att3 = AttentionBlock(F_g=filters[3], F_l=filters[2], n_coefficients=filters[2])
        self.up_conv3 = InceptionConv(filters[3] + filters[2], filters[2])

        self.upsample4 = UpSample(filters[2], filters[2], up_sample_mode='bilinear')
        self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[1], n_coefficients=filters[1])
        self.up_conv4 = InceptionConv(filters[2] + filters[1], filters[1])

        self.upsample5 = UpSample(filters[1], filters[1], up_sample_mode='bilinear')
        self.Att5 = AttentionBlock(F_g=filters[1], F_l=filters[0], n_coefficients=filters[0])
        self.up_conv5 = InceptionConv(filters[1] + filters[0], filters[0])

        self.output_layer = nn.Conv2d(filters[0], output_ch, 1, 1)

    def forward(self, x):
        # Encode
        x1 = self.conv_1(x)
        e1 = self.MaxPool(x1)

        x2 = self.conv_2(e1)
        e2 = self.MaxPool(x2)

        x3 = self.conv_3(e2)
        e3 = self.MaxPool(x3)

        x4 = self.conv_4(e3)
        e4 = self.MaxPool(x4)

        x5 = self.conv_5(e4)
        e5 = self.MaxPool(x5)

        x6 = self.conv_6(e5)

        # Decode
        x66 = self.upsample1(x6)
        g_conv5 = self.Att1(x6, x5)
        x7 = torch.cat((g_conv5, x66), dim=1)
        x8 = self.up_conv1(x7)

        x88 = self.upsample2(x8)
        g_conv4 = self.Att2(x8, x4)
        x9 = torch.cat((g_conv4, x88), dim=1)
        x10 = self.up_conv2(x9)

        x1010 = self.upsample3(x10)
        g_conv3 = self.Att3(x10, x3)
        x11 = torch.cat((g_conv3, x1010), dim=1)
        x12 = self.up_conv3(x11)

        x1212 = self.upsample4(x12)
        g_conv2 = self.Att4(x12, x2)
        x13 = torch.cat((g_conv2, x1212), dim=1)
        x14 = self.up_conv4(x13)

        x1414 = self.upsample5(x14)
        g_conv1 = self.Att5(x14, x1)
        x15 = torch.cat((g_conv1, x1414), dim=1)
        x16 = self.up_conv5(x15)

        output = self.output_layer(x16)

        return output
