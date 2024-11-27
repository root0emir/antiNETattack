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
import joblib
import matplotlib.pyplot as plt
import os

# Veri seti yükleme ve işleme
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # 'protocol' sütununu etiket kodlaması ile dönüştürme
    protocol_encoder = LabelEncoder()
    df['protocol'] = protocol_encoder.fit_transform(df['protocol'])

    # 'attack_type' etiketlerini kodlamak için LabelEncoder kullanma
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df['attack_type'])
    y_one_hot = to_categorical(y_encoded)

    # Özellikler (X) ve Etiketler (y) ayırma
    X = df.drop('attack_type', axis=1).values  # Özellikler
    X = MinMaxScaler().fit_transform(X)  # Veriyi normalleştir

    return X, y_one_hot, encoder

# Model oluşturma
def build_model(input_shape, num_classes):
    model = models.Sequential()

    # Konvolüsyonel katmanlar
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

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Eğitim ve değerlendirme
def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    return history

def evaluate_model(model, X_test, y_test, encoder, history):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Eğitim doğruluğu grafiği
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Eğitim kaybı grafiği
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Performans raporu
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Sınıf tahminlerini al
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

# Ana fonksiyon
def main():
    X, y, encoder = load_and_preprocess_data('network_attack_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Veriyi konvolüsyonel katman için uygun şekilde şekillendir
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = None
    if os.path.exists('./cnn_attack_model.h5'):
        model = models.load_model('./cnn_attack_model.h5')
        print("Model başarıyla yüklendi!")
    else:
        model = build_model(X_train.shape[1:], y.shape[1])
        print("Yeni model oluşturuluyor...")

    model.summary()

    if not os.path.exists('./cnn_attack_model.h5'):
        history = train_model(model, X_train, y_train, X_test, y_test)
        model.save('./cnn_attack_model.h5')

    evaluate_model(model, X_test, y_test, encoder, history)

if __name__ == '__main__':
    main()
