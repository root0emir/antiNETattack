from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from scapy.all import sniff, IP, TCP, UDP
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import logging
from pymongo import MongoClient
from datetime import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import os
from dotenv import load_dotenv

# Çevre değişkenlerini yükleme (API_TOKEN ve MONGO_URI)
load_dotenv()

# Uygulama başlatma
app = FastAPI()

# Kimlik doğrulama için token sistemi
security = HTTPBearer()

# MongoDB bağlantısı
try:
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    db = client["network_logs"]
    collection = db["logs"]
    logging.info("MongoDB bağlantısı başarılı")
except Exception as e:
    logging.error(f"MongoDB bağlantı hatası: {e}")
    raise HTTPException(status_code=500, detail="MongoDB bağlantısı sağlanamadı")

# Model dosya yolları
model_paths = {
    'cnn_model': 'cnn_attack_model.h5',
    'rf_model': 'random_forest_attack_model.pkl',
    'scaler': 'scaler.pkl'
}

# Dosya yollarını kontrol etme
for model_name, model_path in model_paths.items():
    if not os.path.exists(model_path):
        logging.error(f"{model_name} dosyası bulunamadı: {model_path}")
        raise HTTPException(status_code=500, detail=f"{model_name} dosyası yüklenemedi")

# Eğitimli modellerin yüklenmesi
try:
    cnn_model = load_model('cnn_attack_model.h5')  # CNN modeli
    rf_model = joblib.load('random_forest_attack_model.pkl')  # Random Forest modeli
    scaler = joblib.load('scaler.pkl')  # Ölçekleme nesnesi
    logging.info("Modeller başarıyla yüklendi")
except Exception as e:
    logging.error(f"Model yükleme hatası: {e}")
    raise HTTPException(status_code=500, detail=f"Model yükleme hatası: {str(e)}")

# Loglama sistemi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Paket analiz ve özellik çıkarımı
def extract_features(packet):
    try:
        # Paket özelliklerini çıkarma
        packet_size = len(packet)  # Paket boyutu
        protocol = 1 if packet.haslayer(TCP) else 2 if packet.haslayer(UDP) else 0  # Protokol türü
        src_ip = packet[IP].src if packet.haslayer(IP) else "0.0.0.0"  # Kaynak IP
        dst_ip = packet[IP].dst if packet.haslayer(IP) else "0.0.0.0"  # Hedef IP
        timestamp = datetime.now()  # Paket zaman damgası

        features = np.array([[packet_size, protocol]])  # Özellikler
        return features, src_ip, dst_ip, timestamp
    except Exception as e:
        logging.error(f"Paket analizinde hata: {str(e)}")
        raise HTTPException(status_code=500, detail="Paket analizinde hata oluştu")

# CNN ile saldırı türünü tahmin etme
def cnn_predict_attack(features):
    try:
        features = scaler.transform(features)  # Özellikleri normalleştir
        features = np.expand_dims(features, axis=0)  # Modelin girdi şekline uyacak şekilde genişlet
        prediction = cnn_model.predict(features)  # Tahmin yap
        attack_type = np.argmax(prediction)
        attack_labels = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}
        return attack_labels.get(attack_type, "Bilinmiyor")
    except Exception as e:
        logging.error(f"CNN model hatası: {str(e)}")
        return "Hata"

# Random Forest ile belirli saldırı türünü tahmin etme
def rf_predict_attack(features):
    try:
        features = scaler.transform(features)  # Özellikleri normalleştir
        prediction = rf_model.predict(features)  # Tahmin yap
        specific_attack_labels = {0: "Normal", 1: "SYN Flood", 2: "DNS Spoofing", 3: "Port Scanning"}
        return specific_attack_labels.get(prediction[0], "Bilinmiyor")
    except Exception as e:
        logging.error(f"Random Forest model hatası: {str(e)}")
        return "Hata"

# MongoDB'ye log kaydı ekleme
def log_to_mongo(log_entry):
    try:
        collection.insert_one(log_entry)
        logging.info(f"Log MongoDB'ye kaydedildi: {log_entry}")
    except Exception as e:
        logging.error(f"MongoDB kaydetme hatası: {str(e)}")

# Paket verisi için Pydantic modeli
class PacketData(BaseModel):
    packet_data: List[int]

# Ağ izleme endpoint'i
@app.post("/monitor")
async def monitor_network(packet_data: PacketData, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Kimlik doğrulama kontrolü
    if credentials.credentials != os.getenv("API_TOKEN", "API_TOKEN"):
        raise HTTPException(status_code=401, detail="Geçersiz API Token")

    try:
        # Paket verilerini işleme
        features, src_ip, dst_ip, timestamp = extract_features(packet_data.packet_data)

        # CNN ve Random Forest ile saldırı türlerini tahmin etme
        cnn_attack_type = cnn_predict_attack(features)
        rf_attack_type = rf_predict_attack(features)

        # Log kaydı için veri hazırlama
        log_entry = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "packet_size": features[0][0],
            "protocol": features[0][1],
            "timestamp": timestamp,
            "cnn_attack_type": cnn_attack_type,
            "rf_attack_type": rf_attack_type
        }

        # MongoDB'ye log kaydı ekleme
        log_to_mongo(log_entry)

        # Şüpheli etkinlik algılama
        alert_message = "Şüpheli etkinlik tespit edildi" if cnn_attack_type != "Normal" or rf_attack_type != "Normal" else "Trafik normal"

        return {"alert": alert_message, "details": log_entry}

    except Exception as e:
        logging.error(f"Genel hata: {str(e)}")
        raise HTTPException(status_code=500, detail="İç sunucu hatası")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
