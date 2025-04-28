# Residual CNN with side‑to‑move & castling and 4352‑move policy head
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class ChessNet(nn.Module):
    def __init__(self, input_channels: int = 19,
                 num_filters: int = 128,
                 num_res_blocks: int = 40):
        super().__init__()
        self.initial_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.initial_bn   = nn.BatchNorm2d(num_filters)
        self.res_blocks   = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy head  — 4352 logits
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, 4352)

        # Value head
        self.value_conv  = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))
        return p, v
