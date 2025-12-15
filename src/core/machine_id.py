import hashlib, json, uuid, psutil, base58

def cpu_name():
    # Linux / Windows / macOS ortak
    try:
        return psutil.cpu_freq().current
    except:
        return "unknown"

def disk_serial():
    # Basit örnek – gerçekte wmi / hdparm kullanılabilir
    try:
        with open("/etc/machine-id","r") as f:
            return f.read().strip()[:8]
    except:
        return uuid.uuid4().hex[:8]

def tpm_digest():
    return None  # Yoksa None

def ai_fingerprint() -> str:
    raw = {
        "cpu": psutil.cpu_count(logical=False),
        "cpu_name": cpu_name(),
        "mac": uuid.getnode(),
        "disk_serial": disk_serial(),
        "tpm": tpm_digest() or "absent"
    }
    vector = json.dumps(raw, sort_keys=True).encode()
    digest = hashlib.blake2b(vector, digest_size=16).digest()
    return base58.b58encode(digest).decode()