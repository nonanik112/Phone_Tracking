import torch, io
from ai_license import AILicense

class LockedModel(torch.nn.Module):
    def __init__(self, license_code: str, ai_license: AILicense):
        super().__init__()
        self.license = license_code
        self.ai = ai_license
        self._weights = self._decrypt_weights()

    def _decrypt_weights(self):
        try:
            fernet = self.ai._fernet_from_license(self.license)  # basit yardımcı
            with open("src/models/weights.pt.enc", "rb") as f:
                return torch.load(io.BytesIO(fernet.decrypt(f.read())))
        except Exception:
            raise RuntimeError("❌ Geçersiz lisans veya yetkisiz kopya!")

    def forward(self, x):
        return self._weights(x)
