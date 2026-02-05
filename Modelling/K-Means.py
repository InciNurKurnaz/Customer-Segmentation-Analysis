import pandas as pd
import time
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from warnings import filterwarnings

filterwarnings('ignore')

# YAPILANDIRMA VE DOSYA YOLLARI
CLUSTERING_DATA_PATH = '../data/preprocessed/X_clustering_pca_customer.pkl'
# Optimal küme sayısı 2 olarak belirlendi
N_CLUSTERS = 2

# Kayıt klasörlerini tanımla
MODEL_DIR = '../Modelling/models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# YARDIMCI FONKSİYONLAR
def calculate_clustering_metrics(X, labels):
    """Kümeleme sonuçları için Silhouette ve DB Index metriklerini hesaplar."""
    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return {'Silhouette': silhouette, 'DB_Index': db_index}


def run_kmeans(data_path=CLUSTERING_DATA_PATH, k=N_CLUSTERS):
    """
    K-Means algoritmasını uygular, belirlenen K değerini kullanır ve sonuçları kaydeder.
    """
    print("=" * 70)
    print(f"--- K-MEANS KÜMELEME BAŞLATILIYOR (K={k}) ---")
    print("=" * 70)

    # VERİ YÜKLEME
    try:
        X = pd.read_pickle(data_path)
        print(f"✓ PCA uygulanmış Müşteri Verisi (X) başarıyla yüklendi. Boyut: {X.shape}")
    except FileNotFoundError:
        print(f"✗ HATA: Veri dosyası '{data_path}' yolunda bulunamadı.")
        return

    # MODEL EĞİTİMİ VE SÜRE ÖLÇÜMÜ
    start_time = time.time()

    # K-Means Modeli Oluşturma (Analiz edilen K=2 kullanılıyor)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nEğitim Süresi: {training_time:.4f} saniye")


    # PERFORMANS DEĞERLENDİRMESİ

    # 1. Küme Etiketlerini Alma
    labels = kmeans.labels_

    # 2. Metrikleri Hesaplama
    metrics = calculate_clustering_metrics(X, labels)

    print("\n--- K-Means Metrikleri ---")
    print(f"WCSS (Inertia): {kmeans.inertia_:,.2f}")
    print(f"Silhouette Skoru: {metrics['Silhouette']:.4f}")
    print(f"DB Index: {metrics['DB_Index']:.4f}")

    # Küme Büyüklüklerini Kontrol Etme
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    print("\nKüme Büyüklükleri (Segmentasyon Dağılımı):")
    print(cluster_counts.to_string())

    # MODEL VE SONUÇLARI KAYDETME
    # Dosya isimleri müşteri veri setine özgü hale getirildi.
    results_kmeans = {
        'training_time': training_time,
        'metrics': metrics,
        'k': k,
        'wcss': kmeans.inertia_
    }

    # Küme Etiketlerini (labels) orijinal veriye ekleyip analiz için kaydediyorum
    X_labeled = X.copy()
    X_labeled['Cluster'] = labels
    X_labeled.to_pickle(os.path.join(MODEL_DIR, 'X_kmeans_labeled_customer.pkl'))

    # Modeli ve sonuçları kaydet
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'model_kmeans_customer.pkl'))
    joblib.dump(results_kmeans, os.path.join(MODEL_DIR, 'results_kmeans_customer.pkl'))

    print(f"\n[INFO] K-Means modeli ve sonuçları '{MODEL_DIR}' klasörüne başarıyla kaydedildi.")
    print("--- K-MEANS KÜMELEME TAMAMLANDI ---")


if __name__ == "__main__":
    run_kmeans()