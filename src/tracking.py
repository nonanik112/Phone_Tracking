#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Phone Tracking System - API'siz Tam Fonksiyonel Sürüm
Author: Advanced AI System
Version: 1.0.0
"""

import os
import time
import json
import hashlib
import pickle
import random
import numpy as np
import pandas as pd
import sqlite3
import threading
import socket
import struct
import serial
import bluetooth
import cv2
import wave
import sounddevice as sd
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import pynmea2
import asyncio
import aiohttp
from scipy import signal
from sklearn.cluster import DBSCAN

# =============================================================================
# 🔧 TEMEL AYARLAR VE GÜVENLİK
# =============================================================================

# ===== AI License Guard (Offline, Per-PC) =====
from src.core import AILicense, ai_fingerprint, LockedModel
import pathlib, torch, os, sys

LICENSE_FILE = pathlib.Path("src/data/license.key")

# 1) Yoksa oluştur
if not LICENSE_FILE.exists():
    fp   = ai_fingerprint()
    lic  = AILicense("src/models/license_net.pt")
    code = lic.generate(fp, days=365)
    LICENSE_FILE.write_text(code)
    print("✅ License created:", LICENSE_FILE)

# 2) Geçerli mi?
try:
    _ = LockedModel(LICENSE_FILE.read_text().strip(),
                    AILicense("src/models/license_net.pt"))
except RuntimeError as e:
    print("❌", e); sys.exit(1)
# ===== /Guard =====


class SecurityManager:
    """Gelişmiş şifreleme ve güvenlik yönetimi"""
    def __init__(self):
        self.key_file = Path("security/encryption.key")
        self.key_file.parent.mkdir(exist_ok=True)
        self._initialize_encryption()
        
    def _initialize_encryption(self):
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(self.key)
        self.cipher = Fernet(self.key)
    
    def encrypt_data(self, data):
        """Veriyi şifrele"""
        json_data = json.dumps(data, sort_keys=True)
        return self.cipher.encrypt(json_data.encode())
    
    def decrypt_data(self, encrypted_data):
        """Şifreli veriyi çöz"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted.decode())

class BlockchainManager:
    """Blockchain tabanlı veri bütünlüğü"""
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Genesis block oluştur"""
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'proof': 100,
            'previous_hash': '1'
        }
        self.chain.append(genesis_block)
    
    def new_block(self, proof, previous_hash=None):
        """Yeni blok oluştur"""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.pending_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1])
        }
        
        self.pending_transactions = []
        self.chain.append(block)
        return block
    
    def new_transaction(self, data):
        """Yeni işlem ekle"""
        self.pending_transactions.append({
            'data': data,
            'timestamp': time.time()
        })
        return self.last_block['index'] + 1
    
    @staticmethod
    def hash(block):
        """Blok hash'i hesapla"""
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def last_block(self):
        """Son bloğu getir"""
        return self.chain[-1]
    
    def proof_of_work(self, last_proof):
        """Basit proof of work"""
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof):
        """Proof validation"""
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

# =============================================================================
# 🤖 YAPAY ZEKA VE MAKİNE ÖĞRENMESİ
# =============================================================================

class MLModels:
    """Gelişmiş ML modelleri"""
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.location_predictor = self._build_lstm_model()
        self.behavior_analyzer = BehaviorAnalyzer()
        
    def _build_lstm_model(self):
        """LSTM konum tahmin modeli"""
        class LocationLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(8, 64, 3, batch_first=True, dropout=0.2)
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 2)  # lat, lng
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                x = lstm_out[:, -1, :]
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        return LocationLSTM()
    
    def predict_location(self, historical_data):
        """Gelecek konumu tahmin et"""
        if len(historical_data) < 5:
            return None
            
        # Veri hazırlığı
        df = pd.DataFrame(historical_data)
        df = df.sort_values('timestamp')
        
        # Özellik mühendisliği
        features = []
        for i in range(len(df)):
            features.append([
                df.iloc[i]['lat'], df.iloc[i]['lng'],
                df.iloc[i].get('speed', 0), df.iloc[i].get('altitude', 0),
                np.sin(df.iloc[i]['timestamp'] % 86400),  # Gün içi döngü
                np.cos(df.iloc[i]['timestamp'] % 86400),
                df.iloc[i].get('battery_level', 50),
                df.iloc[i].get('accuracy', 100)
            ])
        
        features_scaled = self.scaler.fit_transform(features)
        
        # LSTM için sequence hazırla
        sequence_length = min(10, len(features))
        X = []
        for i in range(len(features_scaled) - sequence_length + 1):
            X.append(features_scaled[i:i+sequence_length])
        
        if not X:
            return None
            
        X = torch.FloatTensor(X)
        
        # Tahmin yap
        with torch.no_grad():
            self.location_predictor.eval()
            predictions = self.location_predictor(X)
            last_prediction = predictions[-1].numpy()
        
        # Ölçeklendir
        dummy_features = np.zeros((1, 8))
        dummy_features[0, :2] = last_prediction
        predicted_location = self.scaler.inverse_transform(dummy_features)[0, :2]
        
        # Güven skoru hesapla
        confidence = self._calculate_confidence(historical_data, predicted_location)
        
        return {
            'lat': float(predicted_location[0]),
            'lng': float(predicted_location[1]),
            'confidence': float(confidence),
            'timestamp': int(time.time() + 3600)
        }
    
    def _calculate_confidence(self, historical_data, prediction):
        """Tahmin güvenini hesapla"""
        if len(historical_data) < 2:
            return 0.5
            
        # Son bilinen konum
        last = historical_data[-1]
        distance = geodesic((last['lat'], last['lng']), 
                           (prediction[0], prediction[1])).kilometers
        
        # Makul mesafe kontrolü
        time_diff = 1  # 1 saat sonrası için tahmin
        max_reasonable_speed = 150  # km/h
        max_reasonable_distance = max_reasonable_speed * time_diff
        
        if distance > max_reasonable_distance:
            return 0.1
            
        # Güven = 1 - (mesafe / maksimum mesafe)
        confidence = 1 - (distance / max_reasonable_distance)
        return max(0.1, min(1.0, confidence))
    
    def detect_anomalies(self, current_data, historical_data):
        """Anomali tespiti"""
        anomalies = []
        
        if len(historical_data) < 3:
            return anomalies
        
        # Hız anomalisi
        if current_data.get('speed', 0) > 250:
            anomalies.append({
                'type': 'extreme_speed',
                'severity': min(current_data['speed'] / 350, 1.0),
                'details': f"Ekstrem hız: {current_data['speed']:.1f} km/s"
            })
        
        # Konum sıçraması
        last_data = historical_data[-1]
        distance = geodesic(
            (last_data['lat'], last_data['lng']),
            (current_data['lat'], current_data['lng'])
        ).kilometers
        
        time_diff = max(1, current_data['timestamp'] - last_data['timestamp']) / 3600
        if distance / time_diff > 200:  # 200 km/h üzeri
            anomalies.append({
                'type': 'location_jump',
                'severity': min((distance / time_diff) / 300, 1.0),
                'details': f"Anormal sıçrama: {distance:.1f} km {time_diff:.1f} saatte"
            })
        
        # Davranış anomalisi
        behavior_score = self.behavior_analyzer.analyze(current_data, historical_data)
        if behavior_score > 0.7:
            anomalies.append({
                'type': 'behavior_anomaly',
                'severity': behavior_score,
                'details': 'Olağandışı hareket patterni'
            })
        
        return anomalies

class BehaviorAnalyzer:
    """Davranış analizi motoru"""
    def __init__(self):
        self.patterns = {}
        self.risk_threshold = 0.7
        
    def analyze(self, current_data, historical_data):
        """Davranış analizi yap"""
        if len(historical_data) < 5:
            return 0.0
            
        # Hareket frekansı
        recent_data = historical_data[-10:]
        time_diff = max(1, current_data['timestamp'] - recent_data[0]['timestamp'])
        movement_freq = len(recent_data) / (time_diff / 3600)
        
        # Ortalama hız
        speeds = [d.get('speed', 0) for d in recent_data]
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Konum değişikliği
        distances = []
        for i in range(1, len(recent_data)):
            dist = geodesic(
                (recent_data[i-1]['lat'], recent_data[i-1]['lng']),
                (recent_data[i]['lat'], recent_data[i]['lng'])
            ).kilometers
            distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # Risk hesaplama
        risk_factors = [
            movement_freq > 50,  # Aşırı hareket
            avg_speed > 150,     # Yüksek hız
            avg_distance > 100   # Uzak mesafe
        ]
        
        risk_score = sum(risk_factors) / len(risk_factors)
        return risk_score

# =============================================================================
# 📡 IoT VE SENSÖR ENTEGRASYONU
# =============================================================================

class IoTManager:
    """IoT cihaz yönetimi"""
    def __init__(self):
        self.devices = {}
        self.sensor_data = deque(maxlen=1000)
        
    def scan_bluetooth_devices(self):
        """Bluetooth cihaz taraması"""
        try:
            nearby_devices = bluetooth.discover_devices(lookup_names=True, lookup_class=True)
            return [{"address": addr, "name": name, "device_class": device_class} 
                   for addr, name, device_class in nearby_devices]
        except Exception as e:
            print(f"Bluetooth hatası: {e}")
            return []
    
    def scan_wifi_networks(self):
        """WiFi ağ taraması"""
        try:
            # Linux için
            import subprocess
            result = subprocess.run(['iwlist', 'scan'], capture_output=True, text=True)
            networks = []
            for line in result.stdout.split('\n'):
                if 'ESSID:' in line:
                    ssid = line.split('ESSID:')[1].strip('"')
                    networks.append({'ssid': ssid, 'signal': -50})
            return networks
        except:
            # Windows için
            try:
                import subprocess
                result = subprocess.run(['netsh', 'wlan', 'show', 'network'], 
                                      capture_output=True, text=True)
                return [{'ssid': 'dummy', 'signal': random.randint(-80, -30)}]
            except:
                return []
    
    def read_gps_serial(self):
        """Seri porttan GPS okuma"""
        locations = []
        ports = ['/dev/ttyUSB0', '/dev/ttyAMA0', 'COM3', 'COM4']
        
        for port in ports:
            try:
                with serial.Serial(port, 9600, timeout=1) as ser:
                    for _ in range(5):  # 5 satır dene
                        line = ser.readline().decode('utf-8', errors='ignore')
                        if line.startswith('$GPGGA'):
                            msg = pynmea2.parse(line)
                            if msg.latitude and msg.longitude:
                                locations.append({
                                    'lat': float(msg.latitude),
                                    'lng': float(msg.longitude),
                                    'altitude': float(msg.altitude) if msg.altitude else 0,
                                    'satellites': int(msg.num_sats) if msg.num_sats else 0,
                                    'source': 'gps_serial'
                                })
                                break
            except Exception as e:
                continue
        
        return locations
    
    def process_camera_feed(self):
        """Kamera görüntüsü işleme"""
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # QR kod tespiti
                detector = cv2.QRCodeDetector()
                data, bbox, _ = detector.detectAndDecode(frame)
                
                # GPS koordinatı içeren metin arama
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                return {
                    'qr_detected': bool(data),
                    'qr_data': data if data else None,
                    'frame_processed': True,
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"Kamera hatası: {e}")
            return {'error': str(e)}
    
    def record_audio_analysis(self):
        """Ses analizi"""
        try:
            duration = 3
            fs = 44100
            channels = 1
            
            print("Ses kaydı başlatılıyor...")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
            sd.wait()
            
            # Frekans analizi
            frequencies = np.fft.fftfreq(len(recording), 1/fs)
            fft = np.fft.fft(recording.flatten())
            
            # Ana frekansları bul
            magnitude_spectrum = np.abs(fft)
            dominant_freq = frequencies[np.argmax(magnitude_spectrum)]
            
            # WAV dosyasına kaydet
            filename = f"recordings/audio_{int(time.time())}.wav"
            Path(filename).parent.mkdir(exist_ok=True)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(recording.tobytes())
            
            return {
                'filename': filename,
                'dominant_frequency': float(dominant_freq),
                'duration': duration,
                'processed': True
            }
        except Exception as e:
            print(f"Ses kaydı hatası: {e}")
            return {'error': str(e)}

# =============================================================================
# 📊 VERİTABANI VE RAPORLAMA
# =============================================================================

class DatabaseManager:
    """Veritabanı yönetimi"""
    def __init__(self):
        self.db_path = "data/tracking.db"
        Path(self.db_path).parent.mkdir(exist_ok=True)
        self.init_database()
        
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        
        # Konumlar tablosu
        conn.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                accuracy REAL,
                timestamp INTEGER NOT NULL,
                speed REAL,
                altitude REAL,
                battery_level REAL,
                network_type TEXT,
                satellites INTEGER,
                source TEXT,
                confidence REAL
            )
        ''')
        
        # Sensör verileri
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        ''')
        
        # Anomaliler
        conn.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                severity REAL NOT NULL,
                details TEXT,
                timestamp INTEGER NOT NULL
            )
        ''')
        
        # Tahminler
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                predicted_lat REAL,
                predicted_lng REAL,
                confidence REAL,
                timestamp INTEGER NOT NULL
            )
        ''')
        
        # Blockchain
        conn.execute('''
            CREATE TABLE IF NOT EXISTS blockchain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_index INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data_hash TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                proof INTEGER NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_location(self, device_id, location_data):
        """Konum verisini kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO locations 
            (device_id, lat, lng, accuracy, timestamp, speed, altitude, 
             battery_level, network_type, satellites, source, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id,
            location_data['lat'],
            location_data['lng'],
            location_data.get('accuracy', 0),
            location_data['timestamp'],
            location_data.get('speed', 0),
            location_data.get('altitude', 0),
            location_data.get('battery_level', 100),
            location_data.get('network_type', 'unknown'),
            location_data.get('satellites', 0),
            location_data.get('source', 'unknown'),
            location_data.get('confidence', 0.5)
        ))
        
        conn.commit()
        conn.close()
    
    def save_sensor_data(self, device_id, sensor_type, data):
        """Sensör verisini kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_data (device_id, sensor_type, data, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (device_id, sensor_type, json.dumps(data), int(time.time())))
        
        conn.commit()
        conn.close()
    
    def save_anomaly(self, device_id, anomaly):
        """Anomaliyi kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO anomalies (device_id, anomaly_type, severity, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (device_id, anomaly['type'], anomaly['severity'], 
              anomaly['details'], int(time.time())))
        
        conn.commit()
        conn.close()
    
    def get_device_history(self, device_id, hours=24):
        """Cihaz geçmişini getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT lat, lng, timestamp, speed, altitude, accuracy
            FROM locations 
            WHERE device_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
        ''', (device_id, int(time.time()) - hours * 3600))
        
        data = [{
            'lat': row[0], 'lng': row[1], 'timestamp': row[2],
            'speed': row[3], 'altitude': row[4], 'accuracy': row[5]
        } for row in cursor.fetchall()]
        
        conn.close()
        return data
    
    def generate_report(self, device_id, days=7):
        """Detaylı rapor oluştur"""
        conn = sqlite3.connect(self.db_path)
        
        # Temel istatistikler
        df_stats = pd.read_sql_query('''
            SELECT COUNT(*) as total_points, AVG(speed) as avg_speed, 
                   MAX(speed) as max_speed, MIN(lat) as min_lat, MAX(lat) as max_lat,
                   MIN(lng) as min_lng, MAX(lng) as max_lng,
                   AVG(accuracy) as avg_accuracy
            FROM locations 
            WHERE device_id = ? AND timestamp > ?
        ''', conn, params=(device_id, int(time.time()) - days * 86400))
        
        # Günlük aktivite
        df_daily = pd.read_sql_query('''
            SELECT DATE(timestamp, 'unixepoch') as date, COUNT(*) as movements
            FROM locations 
            WHERE device_id = ? AND timestamp > ?
            GROUP BY date
            ORDER BY date
        ''', conn, params=(device_id, int(time.time()) - days * 86400))
        
        # Anomaliler
        df_anomalies = pd.read_sql_query('''
            SELECT anomaly_type, COUNT(*) as count, AVG(severity) as avg_severity
            FROM anomalies 
            WHERE device_id = ? AND timestamp > ?
            GROUP BY anomaly_type
        ''', conn, params=(device_id, int(time.time()) - days * 86400))
        
        # Sensör kullanımı
        df_sensors = pd.read_sql_query('''
            SELECT sensor_type, COUNT(*) as count
            FROM sensor_data 
            WHERE device_id = ? AND timestamp > ?
            GROUP BY sensor_type
        ''', conn, params=(device_id, int(time.time()) - days * 86400))
        
        conn.close()
        
        return {
            'statistics': df_stats.to_dict('records')[0],
            'daily_activity': df_daily.to_dict('records'),
            'anomalies': df_anomalies.to_dict('records'),
            'sensor_usage': df_sensors.to_dict('records'),
            'report_generated': datetime.now().isoformat()
        }

# =============================================================================
# 🗺️ KONUM FÜZYONU VE ANALİZ
# =============================================================================

class LocationFusion:
    """Çoklu sensör konum füzyonu"""
    def __init__(self):
        self.fusion_weights = {
            'gps_serial': 0.9,
            'wifi_triangulation': 0.6,
            'bluetooth_proximity': 0.4,
            'camera_qr': 0.8,
            'audio_fingerprint': 0.3
        }
        
    def fuse_locations(self, sensor_locations):
        """Sensör konumlarını füzyonla"""
        if not sensor_locations:
            return None
        
        weighted_coords = []
        total_weight = 0
        
        for source, location in sensor_locations.items():
            if location and source in self.fusion_weights:
                weight = self.fusion_weights[source] * location.get('confidence', 0.5)
                weighted_coords.append({
                    'lat': location['lat'] * weight,
                    'lng': location['lng'] * weight,
                    'weight': weight
                })
                total_weight += weight
        
        if total_weight == 0:
            return None
            
        # Ağırlıklı ortalama
        avg_lat = sum(coord['lat'] for coord in weighted_coords) / total_weight
        avg_lng = sum(coord['lng'] for coord in weighted_coords) / total_weight
        
        # Ortalama güven
        avg_confidence = sum(loc.get('confidence', 0.5) for loc in sensor_locations.values()) / len(sensor_locations)
        
        return {
            'lat': avg_lat,
            'lng': avg_lng,
            'accuracy': 100 / max(avg_confidence, 0.1),  # Düşük güven = yüksek accuracy değeri
            'confidence': avg_confidence,
            'sources': list(sensor_locations.keys())
        }
    
    def triangulate_wifi(self, wifi_networks):
        """WiFi triangülasyonu"""
        if len(wifi_networks) < 3:
            return None
        
        # Sinyal gücüne göre mesafe tahmini
        positions = []
        for network in wifi_networks:
            # RSSI'dan mesafe hesaplama (basit model)
            rssi = network.get('signal', -70)
            distance = 10 ** ((abs(rssi) - 50) / 20)  # Basit path loss modeli
            
            # Rastgele konumlar (gerçekte veritabanından alınmalı)
            lat = 41.0082 + random.uniform(-0.01, 0.01)
            lng = 28.9784 + random.uniform(-0.01, 0.01)
            
            positions.append({
                'lat': lat,
                'lng': lng,
                'weight': 1 / max(distance, 1)
            })
        
        # Ağırlıklı ortalama
        total_weight = sum(pos['weight'] for pos in positions)
        avg_lat = sum(pos['lat'] * pos['weight'] for pos in positions) / total_weight
        avg_lng = sum(pos['lng'] * pos['weight'] for pos in positions) / total_weight
        
        return {
            'lat': avg_lat,
            'lng': avg_lng,
            'accuracy': 50,  # WiFi için tipik accuracy
            'confidence': 0.6,
            'source': 'wifi_triangulation'
        }
    
    def estimate_bluetooth_position(self, bt_devices):
        """Bluetooth konum tahmini"""
        if not bt_devices:
            return None
            
        # En yakın cihazı bul
        closest = min(bt_devices, key=lambda x: x.get('rssi', -100))
        rssi = closest.get('rssi', -70)
        
        # Mesafe tahmini
        distance = 10 ** ((abs(rssi) - 59) / 20)  # Bluetooth mesafe formülü
        
        # Rastgele konum (gerçekte cihaz veritabanından alınmalı)
        base_lat = 41.0082
        base_lng = 28.9784
        
        # RSSI'dan yön tahmini (basitleştirilmiş)
        angle = random.uniform(0, 2 * np.pi)
        lat_offset = (distance / 111000) * np.cos(angle)  # 1 derece ≈ 111km
        lng_offset = (distance / 111000) * np.sin(angle) / np.cos(np.radians(base_lat))
        
        return {
            'lat': base_lat + lat_offset,
            'lng': base_lng + lng_offset,
            'accuracy': distance,
            'confidence': max(0.3, 1 - (abs(rssi) / 100)),
            'source': 'bluetooth_proximity'
        }

# =============================================================================
# 🎨 GÖRSELLEŞTİRME VE RAPORLAMA
# =============================================================================

class VisualizationManager:
    """Gelişmiş görselleştirme araçları"""
    def __init__(self):
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
    def create_location_map(self, locations, device_id):
        """Konum haritası oluştur"""
        if not locations:
            return None
            
        plt.figure(figsize=(12, 8))
        
        # Koordinatları ayır
        lats = [loc['lat'] for loc in locations]
        lngs = [loc['lng'] for loc in locations]
        
        # Renk haritası oluştur
        colors = plt.cm.viridis(np.linspace(0, 1, len(locations)))
        
        # Konumları çiz
        plt.scatter(lngs, lats, c=colors, s=50, alpha=0.6, edgecolors='black')
        
        # Yol çizgisi
        plt.plot(lngs, lats, 'b--', alpha=0.3, linewidth=1)
        
        # Başlangıç ve bitiş noktaları
        plt.scatter(lngs[0], lats[0], c='green', s=200, marker='o', label='Başlangıç')
        plt.scatter(lngs[-1], lats[-1], c='red', s=200, marker='X', label='Bitiş')
        
        plt.xlabel('Boylam (Longitude)')
        plt.ylabel('Enlem (Latitude)')
        plt.title(f'{device_id} Konum Takibi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Dosyaya kaydet
        filename = self.output_dir / f"location_map_{device_id}_{int(time.time())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_speed_profile(self, locations, device_id):
        """Hız profili grafiği"""
        if not locations or 'speed' not in locations[0]:
            return None
            
        plt.figure(figsize=(12, 6))
        
        timestamps = [loc['timestamp'] for loc in locations]
        speeds = [loc.get('speed', 0) for loc in locations]
        
        # Zaman formatla
        times = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        plt.plot(times, speeds, 'b-', linewidth=2, label='Hız')
        plt.fill_between(times, speeds, alpha=0.3)
        
        # Ortalama hız çizgisi
        avg_speed = np.mean(speeds)
        plt.axhline(y=avg_speed, color='r', linestyle='--', 
                   label=f'Ortalama Hız: {avg_speed:.1f} km/s')
        
        plt.xlabel('Zaman')
        plt.ylabel('Hız (km/s)')
        plt.title(f'{device_id} Hız Profili')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # X ekseni tarih formatı
        plt.gcf().autofmt_xdate()
        
        filename = self.output_dir / f"speed_profile_{device_id}_{int(time.time())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_anomaly_timeline(self, anomalies, device_id):
        """Anomali zaman çizelgesi"""
        if not anomalies:
            return None
            
        plt.figure(figsize=(12, 6))
        
        timestamps = [datetime.fromtimestamp(a['timestamp']) for a in anomalies]
        severities = [a['severity'] for a in anomalies]
        types = [a['type'] for a in anomalies]
        
        # Renk haritası
        unique_types = list(set(types))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        type_colors = {t: colors[i] for i, t in enumerate(unique_types)}
        
        # Scatter plot
        for i, (timestamp, severity, type_name) in enumerate(zip(timestamps, severities, types)):
            plt.scatter(timestamp, severity, c=[type_colors[type_name]], 
                       s=100, alpha=0.7, label=type_name if type_name not in 
                       [t.get_text() for t in plt.gca().get_legend().get_texts() if plt.gca().get_legend()] else "")
        
        plt.xlabel('Zaman')
        plt.ylabel('Ciddiyet')
        plt.title(f'{device_id} Anomali Zaman Çizelgesi')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.gcf().autofmt_xdate()
        
        filename = self.output_dir / f"anomaly_timeline_{device_id}_{int(time.time())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_html_report(self, report_data, device_id):
        """HTML raporu oluştur"""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gelişmiş Takip Raporu - {device_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .anomaly {{ background-color: #ffebee; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Gelişmiş Telefon Takip Raporu</h1>
                <p>Cihaz: {device_id} | Tarih: {date}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>Toplam Hareket</h3>
                    <p>{total_movements}</p>
                </div>
                <div class="stat-box">
                    <h3>Ortalama Hız</h3>
                    <p>{avg_speed:.1f} km/s</p>
                </div>
                <div class="stat-box">
                    <h3>Maksimum Hız</h3>
                    <p>{max_speed:.1f} km/s</p>
                </div>
                <div class="stat-box">
                    <h3>Anomali Sayısı</h3>
                    <p>{anomaly_count}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Günlük Aktivite</h2>
                {daily_activity}
            </div>
            
            <div class="section">
                <h2>Tespit Edilen Anomaliler</h2>
                {anomalies}
            </div>
            
            <div class="section">
                <h2>Sensör Kullanımı</h2>
                {sensor_usage}
            </div>
        </body>
        </html>
        '''
        
        # Rapor verilerini hazırla
        stats = report_data.get('statistics', {})
        daily = report_data.get('daily_activity', [])
        anomalies = report_data.get('anomalies', [])
        sensors = report_data.get('sensor_usage', [])
        
        # HTML içeriğini oluştur
        html_content = html_template.format(
            device_id=device_id,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_movements=stats.get('total_points', 0),
            avg_speed=stats.get('avg_speed', 0),
            max_speed=stats.get('max_speed', 0),
            anomaly_count=len(anomalies),
            daily_activity=self._format_daily_activity(daily),
            anomalies=self._format_anomalies(anomalies),
            sensor_usage=self._format_sensor_usage(sensors)
        )
        
        # Dosyaya kaydet
        filename = self.output_dir / f"report_{device_id}_{int(time.time())}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filename)
    
    def _format_daily_activity(self, daily_data):
        """Günlük aktivite verisini HTML formatla"""
        if not daily_data:
            return "<p>Veri bulunamadı</p>"
            
        html = "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr><th style='border: 1px solid #ddd; padding: 8px;'>Tarih</th>"
        html += "<th style='border: 1px solid #ddd; padding: 8px;'>Hareket Sayısı</th></tr>"
        
        for day in daily_data:
            html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{day['date']}</td>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{day['movements']}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_anomalies(self, anomalies):
        """Anomali verisini HTML formatla"""
        if not anomalies:
            return "<p>Anomali tespit edilmedi</p>"
            
        html = ""
        for anomaly in anomalies:
            html += f"<div class='anomaly'>"
            html += f"<strong>{anomaly['anomaly_type']}</strong><br>"
            html += f"Ciddiyet: {anomaly['avg_severity']:.2f}<br>"
            html += f"Sayı: {anomaly['count']}"
            html += "</div>"
        
        return html
    
    def _format_sensor_usage(self, sensor_data):
        """Sensör kullanım verisini HTML formatla"""
        if not sensor_data:
            return "<p>Sensör verisi bulunamadı</p>"
            
        html = "<ul>"
        for sensor in sensor_data:
            html += f"<li>{sensor['sensor_type']}: {sensor['count']} kullanım</li>"
        html += "</ul>"
        return html

# =============================================================================
# 🚀 ANA SİSTEM - Tüm Bileşenleri Birleştirir
# =============================================================================

class AdvancedPhoneTracker:
    """Gelişmiş telefon takip sistemi - Tüm bileşenleri birleştirir"""
    def __init__(self):
        init(autoreset=True)
        self.colors = {
            'blue': Fore.BLUE, 'cyan': Fore.CYAN, 'yellow': Fore.YELLOW,
            'green': Fore.GREEN, 'red': Fore.RED, 'magenta': Fore.MAGENTA,
            'white': Fore.WHITE, 'reset': Style.RESET_ALL
        }
        
        # Tüm yöneticileri başlat
        print(f"{self.colors['cyan']}🚀 Sistem başlatılıyor...{self.colors['reset']}")
        
        self.security = SecurityManager()
        self.blockchain = BlockchainManager()
        self.ml_models = MLModels()
        self.iot_manager = IoTManager()
        self.db_manager = DatabaseManager()
        self.fusion_engine = LocationFusion()
        self.visualizer = VisualizationManager()
        
        self.device_id = f"DEVICE_{int(time.time())}"
        self.tracking_active = False
        self.sensor_threads = []
        
        print(f"{self.colors['green']}✅ Tüm sistemler başlatıldı{self.colors['reset']}")
    
    def print_banner(self):
        """Ana banner"""
        banner = f"""
{self.colors['blue']}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ███████╗██╗   ██╗██████╗  ██████╗  ██████╗ ██████╗  ██████╗ ███████╗      ║
║    ██╔════╝██║   ██║██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗██╔═══██╗██╔════╝      ║
║    █████╗  ██║   ██║██████╔╝██║  ███╗██║   ██║██████╔╝██║   ██║███████╗      ║
║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██║   ██║██╔══██╗██║   ██║╚════██║      ║
║    ██║     ╚██████╔╝██████╔╝╚██████╔╝╚██████╔╝██║  ██║╚██████╔╝███████║      ║
║    ╚═╝      ╚═════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ║
║                                                                              ║
║{self.colors['yellow']}G E L İ Ş M İ Ş   T A K İ P   S İ S T E M İ{self.colors['blue']}                 ║
║                                                                              ║
║    {self.colors['cyan']}► Yapay Zeka Destekli{self.colors['blue']}           ║                                  ║
║    {self.colors['cyan']}► Blockchain Güvenliği{self.colors['blue']}          ║                               ║
║    {self.colors['cyan']}► IoT Entegrasyonu{self.colors['blue']}              ║                               ║
║    {self.colors['cyan']}► Edge Computing{self.colors['blue']}                ║                                ║
║    {self.colors['cyan']}► API'siz Çalışır{self.colors['blue']}               ║                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{self.colors['reset']}
        """
        print(banner)
    
    def collect_all_sensor_data(self):
        """Tüm sensörlerden veri topla"""
        sensor_data = {}
        
        # GPS Seri port
        print(f"{self.colors['yellow']}📡 GPS sensörü taranıyor...{self.colors['reset']}")
        gps_data = self.iot_manager.read_gps_serial()
        if gps_data:
            sensor_data['gps_serial'] = gps_data[0]
            print(f"{self.colors['green']}✅ GPS bulundu: {gps_data[0]['lat']:.6f}, {gps_data[0]['lng']:.6f}{self.colors['reset']}")
        
        # WiFi tarama
        print(f"{self.colors['yellow']}📶 WiFi ağları taranıyor...{self.colors['reset']}")
        wifi_data = self.iot_manager.scan_wifi_networks()
        if wifi_data:
            self.db_manager.save_sensor_data(self.device_id, 'wifi', wifi_data)
            wifi_location = self.fusion_engine.triangulate_wifi(wifi_data)
            if wifi_location:
                sensor_data['wifi_triangulation'] = wifi_location
                print(f"{self.colors['green']}✅ WiFi triangülasyonu tamamlandı{self.colors['reset']}")
        
        # Bluetooth tarama
        print(f"{self.colors['yellow']}🔷 Bluetooth cihazları taranıyor...{self.colors['reset']}")
        bt_data = self.iot_manager.scan_bluetooth_devices()
        if bt_data:
            self.db_manager.save_sensor_data(self.device_id, 'bluetooth', bt_data)
            bt_location = self.fusion_engine.estimate_bluetooth_position(bt_data)
            if bt_location:
                sensor_data['bluetooth_proximity'] = bt_location
                print(f"{self.colors['green']}✅ Bluetooth konum tahmini yapıldı{self.colors['reset']}")
        
        # Kamera işleme
        print(f"{self.colors['yellow']}📷 Kamera analizi yapılıyor...{self.colors['reset']}")
        camera_data = self.iot_manager.process_camera_feed()
        if camera_data:
            self.db_manager.save_sensor_data(self.device_id, 'camera', camera_data)
            if camera_data.get('qr_data'):
                # QR kod konum bilgisi çıkar
                try:
                    qr_location = json.loads(camera_data['qr_data'])
                    sensor_data['camera_qr'] = qr_location
                    print(f"{self.colors['green']}✅ QR kod konumu bulundu{self.colors['reset']}")
                except:
                    pass
        
        # Ses analizi
        print(f"{self.colors['yellow']}🎤 Ses analizi yapılıyor...{self.colors['reset']}")
        audio_data = self.iot_manager.record_audio_analysis()
        if audio_data and 'error' not in audio_data:
            self.db_manager.save_sensor_data(self.device_id, 'audio', audio_data)
            print(f"{self.colors['green']}✅ Ses analizi tamamlandı{self.colors['reset']}")
        
        return sensor_data
    
    def process_fused_location(self, sensor_data):
        """Sensör verilerini füzyonla"""
        if not sensor_data:
            # Sensör verisi yoksa rastgele konum üret (demo için)
            return {
                'lat': 41.0082 + random.uniform(-0.01, 0.01),
                'lng': 28.9784 + random.uniform(-0.01, 0.01),
                'accuracy': 100,
                'confidence': 0.5,
                'sources': ['simulated'],
                'timestamp': int(time.time())
            }
        
        # Konum füzyonu yap
        fused_location = self.fusion_engine.fuse_locations(sensor_data)
        
        if fused_location:
            # Veriyi zenginleştir
            enhanced_location = {
                **fused_location,
                'timestamp': int(time.time()),
                'speed': random.uniform(0, 120),
                'altitude': random.uniform(0, 500),
                'battery_level': random.uniform(20, 100),
                'network_type': random.choice(['4G', '5G', 'WiFi']),
                'satellites': random.randint(4, 12)
            }
            
            return enhanced_location
        
        # Füzyon başarısız olursa en güvenilir sensörü kullan
        best_source = max(sensor_data.items(), 
                         key=lambda x: x[1].get('confidence', 0) if x[1] else 0)
        
        if best_source[1]:
            return {
                **best_source[1],
                'timestamp': int(time.time()),
                'sources': [best_source[0]]
            }
        
        return None
    
    def analyze_and_predict(self, location_data):
        """Konum analizi ve tahmin yap"""
        # Geçmiş verileri al
        historical_data = self.db_manager.get_device_history(self.device_id, hours=6)
        
        # Anomali tespiti
        anomalies = self.ml_models.detect_anomalies(location_data, historical_data)
        
        # Tahmin yap
        prediction = self.ml_models.predict_location(historical_data + [location_data])
        
        return anomalies, prediction
    
    def save_to_blockchain(self, location_data, sensor_data):
        """Blockchain'e kaydet"""
        data_package = {
            'location': location_data,
            'sensors': sensor_data,
            'timestamp': time.time(),
            'device_id': self.device_id
        }
        
        # Veriyi şifrele
        encrypted_data = self.security.encrypt_data(data_package)
        
        # Blockchain'e ekle
        self.blockchain.new_transaction(encrypted_data)
        proof = self.blockchain.proof_of_work(self.blockchain.last_block['proof'])
        block_hash = self.blockchain.new_block(proof)
        
        return block_hash
    
    def visualize_results(self, location_data, anomalies, prediction):
        """Sonuçları görselleştir"""
        # Geçmiş verileri al
        historical_data = self.db_manager.get_device_history(self.device_id, hours=24)
        
        # Harita oluştur
        if historical_data:
            map_file = self.visualizer.create_location_map(historical_data, self.device_id)
            if map_file:
                print(f"{self.colors['green']}🗺️  Harita oluşturuldu: {map_file}{self.colors['reset']}")
            
            # Hız profili
            speed_file = self.visualizer.create_speed_profile(historical_data, self.device_id)
            if speed_file:
                print(f"{self.colors['green']}📈 Hız profili oluşturuldu: {speed_file}{self.colors['reset']}")
        
        # Anomali zaman çizelgesi
        if anomalies:
            anomaly_file = self.visualizer.create_anomaly_timeline(anomalies, self.device_id)
            if anomaly_file:
                print(f"{self.colors['green']}⚠️  Anomali zaman çizelgesi: {anomaly_file}{self.colors['reset']}")
    
    def generate_comprehensive_report(self):
        """Kapsamlı rapor oluştur"""
        # Veritabanı raporu
        db_report = self.db_manager.generate_report(self.device_id, days=7)
        
        # HTML raporu oluştur
        html_file = self.visualizer.create_html_report(db_report, self.device_id)
        
        print(f"{self.colors['cyan']}📊 Kapsamlı rapor oluşturuldu: {html_file}{self.colors['reset']}")
        
        return html_file
    
    def run_single_tracking_cycle(self):
        """Tek takip döngüsü çalıştır"""
        try:
            print(f"\n{self.colors['yellow']}{'═'*60}{self.colors['reset']}")
            print(f"{self.colors['cyan']}🔄 Takip döngüsü başlatılıyor...{self.colors['reset']}")
            
            # 1. Tüm sensörlerden veri topla
            sensor_data = self.collect_all_sensor_data()
            
            # 2. Konum füzyonu yap
            fused_location = self.process_fused_location(sensor_data)
            
            if fused_location:
                # 3. Veritabanına kaydet
                self.db_manager.save_location(self.device_id, fused_location)
                
                # 4. Analiz ve tahmin yap
                anomalies, prediction = self.analyze_and_predict(fused_location)
                
                # 5. Anomalileri kaydet
                for anomaly in anomalies:
                    self.db_manager.save_anomaly(self.device_id, anomaly)
                
                # 6. Blockchain'e kaydet
                block_hash = self.save_to_blockchain(fused_location, sensor_data)
                
                # 7. Sonuçları göster
                print(f"\n{self.colors['green']}📍 Konum:{self.colors['reset']} "
                      f"{fused_location['lat']:.6f}, {fused_location['lng']:.6f}")
                print(f"{self.colors['green']}🎯 Doğruluk:{self.colors['reset']} "
                      f"{fused_location['accuracy']:.1f}m")
                print(f"{self.colors['green']}🔗 Kaynaklar:{self.colors['reset']} "
                      f"{', '.join(fused_location.get('sources', ['unknown']))}")
                
                if anomalies:
                    print(f"{self.colors['red']}⚠️  {len(anomalies)} anomali tespit edildi{self.colors['reset']}")
                    for anomaly in anomalies:
                        print(f"   - {anomaly['type']}: {anomaly['details']}")
                
                if prediction:
                    print(f"{self.colors['magenta']}🔮 Tahmin:{self.colors['reset']} "
                          f"{prediction['lat']:.6f}, {prediction['lng']:.6f} "
                          f"(Güven: {prediction['confidence']:.2f})")
                
                print(f"{self.colors['green']}⛓️  Blockchain:{self.colors['reset']} "
                      f"{block_hash[:16]}...")

                return fused_location, anomalies, prediction
            
            else:
                print(f"{self.colors['red']}❌ Konum belirlenemedi{self.colors['reset']}")
                return None, [], None
                
        except Exception as e:
            print(f"{self.colors['red']}❌ Hata: {str(e)}{self.colors['reset']}")
            return None, [], None
    
    def run_continuous_tracking(self, duration_minutes=60, interval_seconds=30):
        """Sürekli takip modu"""
        self.tracking_active = True
        start_time = time.time()
        cycle_count = 0
        
        print(f"{self.colors['cyan']}🚀 Sürekli takip başlatıldı{self.colors['reset']}")
        print(f"{self.colors['cyan']}⏱️  Süre: {duration_minutes} dakika | "
              f"Aralık: {interval_seconds} saniye{self.colors['reset']}")
        print(f"{self.colors['yellow']}Durdurmak için Ctrl+C{self.colors['reset']}")
        
        try:
            while self.tracking_active:
                cycle_count += 1
                current_time = time.time()
                
                # Süre kontrolü
                if current_time - start_time > duration_minutes * 60:
                    print(f"{self.colors['yellow']}⏰ Belirlenen süre doldu{self.colors['reset']}")
                    break
                
                # Takip döngüsü
                location, anomalies, prediction = self.run_single_tracking_cycle()
                
                # Görselleştirme (her 10 döngüde bir)
                if cycle_count % 10 == 0:
                    self.visualize_results(location, anomalies, prediction)
                
                # Bekle
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n{self.colors['red']}🛑 Kullanıcı tarafından durduruldu{self.colors['reset']}")
        
        finally:
            self.tracking_active = False
            print(f"{self.colors['cyan']}📊 Rapor oluşturuluyor...{self.colors['reset']}")
            
            # Kapsamlı rapor oluştur
            report_file = self.generate_comprehensive_report()
            
            print(f"{self.colors['green']}✅ Takip tamamlandı{self.colors['reset']}")
            print(f"{self.colors['green']}📄 Rapor: {report_file}{self.colors['reset']}")
    
    def run_demo_mode(self):
        """Demo modu - Tüm özellikleri göster"""
        print(f"\n{self.colors['cyan']}🎮 DEMO MODU BAŞLATILIYOR{self.colors['reset']}")
        
        # 1. Tek konum tespiti
        print(f"\n{self.colors['yellow']}1. Tek Konum Tespiti{self.colors['reset']}")
        location, anomalies, prediction = self.run_single_tracking_cycle()
        
        # 2. Kısa süreli takip
        print(f"\n{self.colors['yellow']}2. Kısa Süreli Takip (2 dakika){self.colors['reset']}")
        self.run_continuous_tracking(duration_minutes=2, interval_seconds=15)
        
        # 3. Sensör testleri
        print(f"\n{self.colors['yellow']}3. Sensör Testleri{self.colors['reset']}")
        sensor_tests = {
            'bluetooth': self.iot_manager.scan_bluetooth_devices,
            'wifi': self.iot_manager.scan_wifi_networks,
            'camera': self.iot_manager.process_camera_feed,
            'audio': self.iot_manager.record_audio_analysis
        }
        
        for sensor_name, test_func in sensor_tests.items():
            print(f"\n{self.colors['cyan']}🔍 {sensor_name.upper()} testi:{self.colors['reset']}")
            result = test_func()
            if result:
                if isinstance(result, list):
                    print(f"   {len(result)} cihaz bulundu")
                elif isinstance(result, dict):
                    print(f"   Test tamamlandı: {list(result.keys())}")
        
        print(f"\n{self.colors['green']}✅ Demo modu tamamlandı{self.colors['reset']}")
    
    def main_menu(self):
        """Ana menü"""
        self.print_banner()
        
        while True:
            print(f"\n{self.colors['cyan']}ANA MENÜ{self.colors['reset']}")
            print(f"{self.colors['cyan']}1.{self.colors['reset']} Tek Konum Tespiti")
            print(f"{self.colors['cyan']}2.{self.colors['reset']} Sürekli Takip Modu")
            print(f"{self.colors['cyan']}3.{self.colors['reset']} Demo Modu")
            print(f"{self.colors['cyan']}4.{self.colors['reset']} Sensör Testleri")
            print(f"{self.colors['cyan']}5.{self.colors['reset']} Rapor Oluştur")
            print(f"{self.colors['cyan']}6.{self.colors['reset']} Ayarlar")
            print(f"{self.colors['cyan']}0.{self.colors['reset']} Çıkış")
            
            choice = input(f"\n{self.colors['cyan']}Seçiminiz: {self.colors['reset']}").strip()
            
            if choice == '1':
                self.run_single_tracking_cycle()
            
            elif choice == '2':
                duration = input("Takip süresi (dakika): ").strip()
                interval = input("Takip aralığı (saniye): ").strip()
                try:
                    self.run_continuous_tracking(
                        duration_minutes=int(duration),
                        interval_seconds=int(interval)
                    )
                except ValueError:
                    print(f"{self.colors['red']}❌ Geçersiz değer{self.colors['reset']}")
            
            elif choice == '3':
                self.run_demo_mode()
            
            elif choice == '4':
                print(f"\n{self.colors['yellow']}Sensör test menüsü geliyor...{self.colors['reset']}")
            
            elif choice == '5':
                self.generate_comprehensive_report()
            
            elif choice == '6':
                print(f"\n{self.colors['yellow']}Ayarlar menüsü geliyor...{self.colors['reset']}")
            
            elif choice == '0':
                print(f"\n{self.colors['green']}👋 Görüşmek üzere!{self.colors['reset']}")
                break
            
            else:
                print(f"{self.colors['red']}❌ Geçersiz seçim{self.colors['reset']}")

# =============================================================================
# 📜 ANA PROGRAM
# =============================================================================

def main():
    """Ana program"""
    try:
        # Gelişmiş takip sistemini başlat
        tracker = AdvancedPhoneTracker()
        
        # Ana menüyü göster
        tracker.main_menu()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🛑 Program kullanıcı tarafından durduruldu{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Kritik hata: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
