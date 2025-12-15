echo "from .ai_license import AILicense" > src/core/__init__.py
echo "from .machine_id import ai_fingerprint" >> src/core/__init__.py
echo "from .model_lock import LockedModel" >> src/core/__init__.py

# 2) Şifreli ağırlık placeholder (gerçek modeli sonra koyarsın)
touch src/models/weights.pt.enc

# 3) data klasörü (license.key buraya inecek)
mkdir -p src/data
touch src/data/.gitkeep