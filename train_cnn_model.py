import pandas as pd
import numpy as np 
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import load_model
import os

# Veri seti yükleme ve ön işleme
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Eksik verileri sıfırla dolduruyoruz
    df.fillna(0, inplace=True)

    # Özellikler (X) ve Etiketler (y) ayırma
    X = df.drop('attack_type', axis=1).values  # Özellikler
    y = df['attack_type'].values  # Etiketler

    # Etiketleri dönüştürme (Label Encoding ve One-Hot Encoding)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)

    # Veriyi normalize etme
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_one_hot, encoder, scaler

# Modelin oluşturulması
def build_model(input_shape, num_classes):
    model = models.Sequential()

    # Konvolüsyonel katman
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.2))

    # Daha derin bir ağ
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Yığınlama katmanları
    model.add(layers.Conv1D(256, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Fully Connected katmanlar
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))

    # Çıktı katmanı
    model.add(layers.Dense(num_classes, activation='softmax'))  # Çok sınıflı sınıflandırma için softmax

    # Modeli derleme
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Modeli eğitme
def train_model(model, X_train, y_train, X_test, y_test):
    # Erken durdurma callback'i
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Modeli eğitme
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                        validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    return history

# Performans değerlendirmesi
def evaluate_model(model, X_test, y_test, encoder, history):
    # Modelin değerlendirilmesi
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Eğitim ve doğrulama doğruluğu grafiği
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Eğitim ve doğrulama kaybı grafiği
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Performans raporu
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Sınıf tahminlerini al

    # Gerçek sınıfları geri çevir
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

# Ana fonksiyon
def main():
    # Veri seti yükleme ve ön işleme
    X_scaled, y_one_hot, encoder, scaler = load_and_preprocess_data('network_attack_data.csv')

    # Eğitim ve test verisini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.3, random_state=42)

    # Veriyi konvolüsyonel katmanlar için uygun şekilde şekillendirelim
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Modeli oluşturma
    model = None
    if os.path.exists('./cnn_attack_model.h5'):
        model = load_model('./cnn_attack_model.h5')
        print("Model başarıyla yüklendi!")
    else:
        model = build_model(X_train.shape[1:], y_one_hot.shape[1])
        print("Yeni model oluşturuluyor...")

    # Modelin özetini yazdıralım
    model.summary()

    # Modeli eğitelim (Eğer model zaten yoksa eğitilecek)
    if model is not None and not os.path.exists('./cnn_attack_model.h5'):
        history = train_model(model, X_train, y_train, X_test, y_test)

        # Modelin değerlendirilmesi
        evaluate_model(model, X_test, y_test, encoder, history)

        # Modeli kaydetme
        model.save('cnn_attack_model.h5')

        # Etiket ve scaler'ı kaydedelim
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(encoder, 'encoder.pkl')

# Ana fonksiyonu çalıştırma
if __name__ == "__main__":
    main()
