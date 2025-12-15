import torch, torch.nn as nn

class LicenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, x):
        return self.fc(x)

net = LicenseNet()
traced = torch.jit.trace(net, torch.randn(1, 16))

openssl enc -aes-256-cbc -salt -in src/models/license_net.pt \
          -out src/models/weights.pt.enc \
          -k "$(cat src/data/license.key)"
