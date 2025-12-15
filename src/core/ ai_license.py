# ai_license.py
from cryptography.fernet import Fernet
import torch, base58, datetime, json

class AILicense:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)   # TorchScript
        self.model.eval()

    def generate(self, fingerprint: str, days: int = 365) -> str:
        """
        Giriş: finger-print vektör
        Çıkış: base58 lisans kodu (sadece bu vektör için geçerli)
        """
        vec = torch.tensor([ord(c) for c in fingerprint[:16]], dtype=torch.float32)
        with torch.no_grad():
            key_vec = self.model(vec.unsqueeze(0)).squeeze(0)  # 16 boyut
        key_bytes = bytes((key_vec * 255).byte().tolist())
        fernet = Fernet(base58.b58encode(key_bytes[:32] + b'0' * 8)[:32] + b'=')
        payload = {
            "fp": fingerprint,
            "exp": (datetime.datetime.now() + datetime.timedelta(days=days)).isoformat()
        }
        license_code = fernet.encrypt(json.dumps(payload).encode())
        return base58.b58encode(license_code).decode()