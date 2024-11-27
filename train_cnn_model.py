import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt

# GPU sorunları varsa sadece CPU kullanımı için ayar
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Veri yükleme ve işleme
def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Orijinal Veri Şekli:", df.shape)

        # 'protocol' sütunu varsa dönüştür
        if 'protocol' in df.columns:
            protocol_encoder = LabelEncoder()
            df['protocol'] = protocol_encoder.fit_transform(df['protocol'])
        else:
            print("'protocol' sütunu eksik, atlanıyor...")

        # 'attack_type' sütununu kontrol et ve kodla
        if 'attack_type' not in df.columns:
            raise ValueError("'attack_type' sütunu veri çerçevesinde bulunamadı!")

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(df['attack_type'])
        y_one_hot = to_categorical(y_encoded)

        # Özellikleri seç ve ölçeklendir
        X = df.drop('attack_type', axis=1).values
        if X.shape[1] < 3:
            raise ValueError("Özellik sayısı Conv1D için yetersiz. En az 3 özellik gereklidir.")

        X = MinMaxScaler().fit_transform(X)

        return X, y_one_hot, encoder
    except Exception as e:
        print(f"Veri işleme sırasında hata oluştu: {e}")
        raise

# Model oluşturma
def build_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Conv1D katmanları
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(256, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Dense katmanları
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Çok sınıflı sınıflandırma için softmax

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Model eğitimi
def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])
    return history

# Modeli değerlendirme
def evaluate_model(model, X_test, y_test, encoder, history):
    # Test seti üzerinde performans
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Eğitim ve doğrulama doğruluğunu çiz
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Eğitim ve doğrulama kaybını çiz
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Tahmin ve raporlar
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

# Ana fonksiyon
def main():
    try:
        # Veriyi yükle ve işle
        filepath = 'network_attack_data.csv'
        X, y, encoder = load_and_preprocess_data(filepath)

        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # CNN için veriyi şekillendir
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # 3D giriş
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # 3D giriş

        # Modeli oluştur veya yükle
        model_path = './cnn_attack_model.h5'
        if os.path.exists(model_path):
            model = models.load_model(model_path)
            print("Model başarıyla yüklendi!")
        else:
            model = build_model(X_train.shape[1:], y.shape[1])  # Yeni model oluştur
            print("Yeni model oluşturuluyor...")

        model.summary()

        # Model eğit ve kaydet
        if not os.path.exists(model_path):
            history = train_model(model, X_train, y_train, X_test, y_test)
            model.save(model_path)
            print("Model kaydedildi!")
        else:
            history = None

        # Modeli değerlendir
        if history:
            evaluate_model(model, X_test, y_test, encoder, history)
    except Exception as e:
        print(f"Program sırasında hata oluştu: {e}")

if __name__ == '__main__':
    main()
