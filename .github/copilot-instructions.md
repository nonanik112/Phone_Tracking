<!-- Copilot instructions for contributors and AI coding agents -->
# Copilot / AI Agent Instructions

Kısa: Bu repo `Advanced Phone Tracker` projesinin **hafif** bir versiyonunu içerir. Gerçekte README'de belirtilen modüler yapı (`core/`, `advanced_phone_tracker.py`) büyük ölçüde dokümante edilmiştir, fakat çalışma kodu tek dosya halinde `tracking.py` içindedir. Aşağıda projeye hızlı katkı yaparken bilmeniz gereken önemli, keşfedilebilir noktalar yer alır.

- **Ana çalışma dosyası**: `tracking.py` — tüm sınıflar ve `main()` burada (örnek: `AdvancedPhoneTracker`, `SecurityManager`, `BlockchainManager`, `MLModels`, `IoTManager`). AI ajanları önce bu dosyayı okuyup anlayın.
- **README.md**: Projenin hedef mimarisi, bağımlılıklar ve çalıştırma örnekleri burada; ama bazı dosyalar (ör. `advanced_phone_tracker.py`, `core/`) gerçek repo'da **yok**. Değişiklikler yaparken README'yi güncelleyin.

- **Çalıştırma**: Interaktif ana menü için:

```bash
python tracking.py
```

- **Donanım / Sistem bağımlılıkları**: Kod Bluetooth, seri port, `iwlist` (Linux) ve ses/cv kütüphanelerine dayanır. CI veya otomatik test yazarken gerçek donanım erişimi olmayan durumlar için sensör I/O'larını taklit etmeye (mock) yönelin.

- **Önemli modüller / örnek fonksiyonlar**:
  - `SecurityManager.encrypt_data` / `decrypt_data` — şifreleme anahtarı `security/encryption.key` içinde saklanır.
  - `BlockchainManager.hash`, `proof_of_work`, `valid_proof` — blok zinciri testleri için basit, deterministik fonksiyonlar.
  - `MLModels.predict_location` ve `detect_anomalies` — algoritmik mantık burada; bu kısımlar birim testlerine uygundur.
  - `IoTManager.scan_wifi_networks`, `scan_bluetooth_devices`, `read_gps_serial` — OS/CI bağımlı sensör işlevleri.

- **Test önerileri**: Donanım bağımsız mantık için birim test yazın (örn. `BehaviorAnalyzer.analyze`, `BlockchainManager.hash`, `SecurityManager` şifreleme akışı, `MLModels._calculate_confidence`). Sensör/kaynaklı fonksiyonlar için küçük entegrasyon testlerinde *mock* kullanın.

- **Kod tarzı & katkı notları**:
  - Mevcut kod tek dosyada yoğun; büyük katkılar yaparken modülleri `core/` gibi klasörlere taşımak mantıklı — taşıma yapıldığında `README.md` ve `__main__` davranışını güncelleyin.
  - Sınıf/işlev isimlendirmesi: sınıflar `CamelCase`, fonksiyonlar `snake_case` kullanıyor; mevcut stili devam ettirin.
  - Yeni bağımlılıklar eklemeden önce gerçekten gerekiyorsa onay isteyin (CI/odağı düşünülerek).

- **CI / PR davranışı**:
  - Değişiklikler küçük, atomik ve testlerle gelmeli.
  - Donanım gerektiren değişiklikler için `--sim` veya `--mock` flag'leri ekleyip belgeleyin; CI'de bu mod kullanılmalı.

- **Sık Sorulan Sorular (kısa)**:
  - README'deki dosyalar yoksa ne yapmalı? → Öncelikle `tracking.py`'yi referans alın, sonra modülerleştirme planı hazırlayın ve PR ile önerin.
  - Hangi dosyalar birim test hedefi? → `tracking.py` içindeki algoritmik sınıflar: `MLModels`, `BehaviorAnalyzer`, `BlockchainManager`, `SecurityManager`.

Eğer bu yönergede eksik veya belirsiz bir kısım görürseniz belirtin — isteğinize göre örnek testler veya küçük refactor PR'ları hazırlayabilirim.
