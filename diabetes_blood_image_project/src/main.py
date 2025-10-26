import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 🔹 Sonuç dosyalarına erişim için
from results.metrics.calculate_metrics import save_metrics
from results.logs.logger import log_message
from results.figures.plot_feature_importance import plot_feature_importance


def main():
    log_message("Model eğitimi başlatıldı 🚀")

    # -------------------------------
    # 1️⃣ Veri yükleme
    # -------------------------------
    data_path = "data/features_week1.csv"
    if not os.path.exists(data_path):
        log_message("❌ Veri dosyası bulunamadı.")
        raise FileNotFoundError(f"{data_path} mevcut değil.")
    
    df = pd.read_csv(data_path)
    log_message(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

    X = df.drop(columns=["label"], errors="ignore")
    y = df["label"]

    # -------------------------------
    # 2️⃣ Eğitim / Test bölme
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log_message("Veri eğitim ve test olarak ayrıldı.")

    # -------------------------------
    # 3️⃣ Model eğitimi
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    log_message("Model başarıyla eğitildi.")

    # -------------------------------
    # 4️⃣ Değerlendirme
    # -------------------------------
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report["accuracy"]

    log_message(f"Model doğruluğu: {acc:.4f}")

    # -------------------------------
    # 5️⃣ Sonuçları kaydet
    # -------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/final_rf_v2.joblib"
    joblib.dump(model, model_path)
    log_message(f"Model kaydedildi: {model_path}")

    # 🔹 Metrikleri, grafik ve log dosyalarını kaydet
    save_metrics(y_test, y_pred, model_name="RandomForest_v2")
    plot_feature_importance(model.feature_importances_, X.columns)

    log_message("✅ Tüm sonuçlar başarıyla kaydedildi.")
    print("🎉 Model eğitimi tamamlandı. Sonuçlar 'results/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
