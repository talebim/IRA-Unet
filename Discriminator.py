import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, img_ch=4, output_ch=1, num_fc1=64, filters=[32, 64, 128, 256, 512]):
        super(Network, self).__init__()

        self.dropout_rate = 0.25

        self.conv1 = nn.Conv2d(in_channels=img_ch, out_channels=filters[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[2])
        self.conv5 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.fc1 = nn.Linear(filters[3] * 192 // 16 * 192 // 16, num_fc1)
        self.fc2 = nn.Linear(num_fc1, output_ch)

    def forward(self, input):
        batchsize = input.size()[0]
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = self.pool(output)
        output = F.relu(self.bn5(self.conv5(output)))
        output = self.pool(output)
        output = output.view(batchsize, -1)
        output = F.relu(self.fc1(output))
        output = F.dropout(output, self.dropout_rate)
        output = self.fc2(output)

        return output
