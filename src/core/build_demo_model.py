# build_demo_model.py
import torch, torch.nn as nn, pathlib, subprocess, os

class LicenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, x):
        return self.fc(x)

net = LicenseNet()
traced = torch.jit.trace(net, torch.randn(1, 16))
pathlib.Path("src/models").mkdir(parents=True, exist_ok=True)
traced.save("src/models/license_net.pt")

# Opsiyonel: hemen şifrele (isteğe bağlı)
key = os.getenv("LICENSE_KEY") or "default-fallback-key"
sub.run([
    "openssl", "enc", "-aes-256-cbc", "-salt",
    "-in", "src/models/license_net.pt",
    "-out", "src/models/weights.pt.enc",
    "-k", key
])
