# 📍 Advanced Phone Tracker

> **AI + Blockchain + IoT destekli, çevrimdışı çalışabilen gelişmiş konum takip ve analiz aracı**

![Banner](docs/images/banner.png)

---

## 🚀 Genel Bakış

**Advanced Phone Tracker**, Python ile geliştirilmiş; çoklu sensör füzyonu, yapay zeka destekli analiz ve blockchain tabanlı veri bütünlüğünü tek bir mimaride birleştiren ileri seviye bir konum takip sistemidir.

* 🌐 **İnternetsiz (offline) çalışır**
* ⚡ **Edge Computing** ile düşük gecikme
* 🔐 **Uçtan uca şifreleme**
* 📊 **Otomatik raporlama & görselleştirme**

---

## 🧩 Mimari Genel Görünüm

![Architecture](docs/images/architecture.png)

```text
Sensörler → Füzyon Katmanı → AI Analiz → Blockchain Kayıt → SQLite DB → Raporlama
```

---

## 🛠️ Modüller ve Teknik Detaylar

| Modül                 | Teknik Detay                                              | Açıklama                                                 |
| --------------------- | --------------------------------------------------------- | -------------------------------------------------------- |
| **🧠 Yapay Zeka**     | LSTM + IsolationForest                                    | Gelecek konum tahmini, anomali tespiti, davranış analizi |
| **⛓️ Blockchain**     | SHA-256, Proof-of-Work                                    | Değiştirilemez konum kaydı, veri bütünlüğü               |
| **📡 IoT Füzyonu**    | GPS, Wi‑Fi, Bluetooth, Kamera, Ses                        | Çoklu sensörden tek doğruluklu konum                     |
| **⚡ Edge Computing**  | Lokal işlem, ~100 ms gecikme                              | API’siz, hızlı, çevrimdışı çalışabilir                   |
| **🔐 Güvenlik**       | AES-256 (Fernet)                                          | Konum verisi uçtan uca şifreli                           |
| **🗃️ Veritabanı**    | SQLite + WAL                                              | 10M+ kayıt, indeksli, raporlama hazır                    |
| **📊 Görselleştirme** | Matplotlib + Seaborn                                      | Harita, hız grafiği, anomali zaman çizelgesi             |
| **📄 Raporlama**      | Otomatik HTML + PNG                                       | 7 günlük detaylı rapor, mail uyumlu                      |
| **🔌 Sensörler**      | GPS, Wi‑Fi triangulation, BT proximity, QR Kamera, Ses FP | Gerçek donanım okuması                                   |
| **🤖 Otomasyon**      | threading + asyncio                                       | 30 sn döngü, CPU dostu                                   |
| **💰 Maliyet**        | 0 $                                                       | MIT Lisansı, sınırsız kullanım                           |

---

## 📦 Kurulum

### 1️⃣ Python Bağımlılıkları

```bash
pip install numpy pandas scikit-learn torch cryptography colorama \
            opencv-python pillow sounddevice matplotlib seaborn \
            geopy pyserial pybluez wifi scipy aiohttp
```

### 2️⃣ Linux Sistem Paketleri

```bash
sudo apt-get install bluetooth libbluetooth-dev
```

---

## ▶️ Çalıştırma

```bash
python advanced_phone_tracker.py
```

---

## 🎮 Örnek Kullanım Senaryoları

### 🧪 1. Demo Modu

Tüm özellikleri tek seferde test eder.

```bash
python advanced_phone_tracker.py --demo
```

---

### ⏱️ 2. Sürekli Takip

30 dakika boyunca, her **15 saniyede** bir konum kaydı alır.

```bash
python advanced_phone_tracker.py --track --duration 30 --interval 15
```

---

### 🔍 3. Sensör Testleri

Tüm sensörleri tek tek doğrular.

```bash
python advanced_phone_tracker.py --sensor-test
```

![Sensors](docs/images/sensors.png)

---

### 📑 4. Rapor Oluşturma

Otomatik HTML + PNG rapor üretir.

```bash
python advanced_phone_tracker.py --report
```

![Report](docs/images/report.png)

---

## 📊 Üretilen Çıktılar

* 📍 Konum haritası (PNG)
* 📈 Hız & zaman grafikleri
* 🚨 Anomali zaman çizelgesi
* 📄 HTML dashboard raporu

---

## 🔐 Güvenlik Mimarisi

![Security](docs/images/security.png)

* AES‑256 Fernet şifreleme
* Lokal anahtar üretimi
* Blockchain hash zinciri
* Değiştirilemez kayıtlar

---

## 📁 Proje Yapısı

```text
advanced_phone_tracker/
├── advanced_phone_tracker.py
├── core/
│   ├── ai_engine.py
│   ├── sensor_fusion.py
│   ├── blockchain.py
│   └── security.py
├── reports/
├── database/
├── docs/images/
└── README.md
```

---

## 📜 Lisans

Bu proje **MIT License** ile lisanslanmıştır.

> Tamamen ücretsizdir. Ticari ve kişisel kullanıma açıktır.

---

## 👤 Geliştirici Notu

Bu proje **yüksek gizlilik**, **offline çalışma** ve **gerçek sensör verisi** odaklı tasarlanmıştır. Simülasyon veya üçüncü parti API bağımlılığı yoktur.

---

⭐ Eğer projeyi beğendiysen yıldızlamayı unutma!
