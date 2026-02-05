import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from warnings import filterwarnings

filterwarnings('ignore')


CLUSTERING_DATA_PATH = '../data/preprocessed/X_clustering_pca_customer.pkl'
MODEL_DIR = '../Modelling/models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#DBSCAN PARAMETRELERİ
DBSCAN_EPS = 2.6  # K-Distance Grafiği ile belirlendi
DBSCAN_MIN_SAMPLES = 44  # 2 * 22 Özellik Sayısı


def calculate_clustering_metrics(X, labels):
    """Kümeleme sonuçları için Silhouette ve DB Index metriklerini hesaplar (Gürültü hariç)."""
    # Gürültü noktalarını (-1 etiketli) filtrele
    valid_indices = labels != -1
    X_valid = X[valid_indices]
    labels_valid = labels[valid_indices]

    unique_clusters = np.unique(labels_valid)

    # Geçerli küme sayısı 2'den azsa metrik hesaplanamaz.
    if unique_clusters.size < 2:
        # Tek bir küme veya sadece gürültü varsa
        if unique_clusters.size == 1:
            # Sadece tek bir küme varsa, metrik hesaplanamaz.
            return {'Silhouette': -1.0, 'DB_Index': 999.0}
        # Hiç geçerli küme kalmadıysa
        return {'Silhouette': -1.0, 'DB_Index': 999.0}

    silhouette = silhouette_score(X_valid, labels_valid)
    db_index = davies_bouldin_score(X_valid, labels_valid)
    return {'Silhouette': silhouette, 'DB_Index': db_index}


def run_dbscan(data_path=CLUSTERING_DATA_PATH, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """
    DBSCAN algoritmasını optimize edilmiş parametrelerle uygular ve sonuçları kaydeder.
    """
    print("=" * 70)
    print(f"--- DBSCAN KÜMELEME BAŞLATILIYOR (eps={eps}, min_samples={min_samples}) ---")
    print("=" * 70)

    try:
        X = pd.read_pickle(data_path)
        print(f"✓ PCA uygulanmış Müşteri Verisi (X) başarıyla yüklendi. Boyut: {X.shape}")
    except FileNotFoundError:
        print(f"✗ HATA: Veri dosyası '{data_path}' yolunda bulunamadı.")
        return

    # MODEL EĞİTİMİ
    start_time = time.time()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nEğitim Süresi: {training_time:.4f} saniye")

    # PERFORMANS DEĞERLENDİRMESİ

    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters_ = len([l for l in unique_labels if l != -1])

    # Gürültü sayısını bul
    n_noise_ = 0
    if -1 in unique_labels:
        n_noise_ = counts[unique_labels == -1][0]

    metrics = calculate_clustering_metrics(X.values, labels)

    print("\n--- DBSCAN Sonuçları ve Metrikler ---")
    print(f"Bulunan Küme Sayısı: {n_clusters_}")
    print(f"Gürültü Noktası Sayısı (-1 Etiketli): {n_noise_}")
    print(f"Silhouette Skoru (Gürültü Hariç): {metrics['Silhouette']:.4f}")
    print(f"DB Index (Gürültü Hariç): {metrics['DB_Index']:.4f}")

    # Küme Büyüklüklerini Kontrol Etme
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    print("\nKüme Büyüklükleri (Segmentasyon Dağılımı):")
    print(cluster_counts.to_string())

    # MODEL VE SONUÇLARI KAYDETME
    results_dbscan = {
        'training_time': training_time,
        'metrics': metrics,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters_,
        'n_noise': n_noise_
    }

    X_labeled = X.copy()
    X_labeled['Cluster_DBSCAN'] = labels
    X_labeled.to_pickle(os.path.join(MODEL_DIR, 'X_dbscan_labeled_customer.pkl'))

    joblib.dump(dbscan, os.path.join(MODEL_DIR, 'model_dbscan_customer.pkl'))
    joblib.dump(results_dbscan, os.path.join(MODEL_DIR, 'results_dbscan_customer.pkl'))

    print(f"\n[INFO] DBSCAN modeli ve sonuçları '{MODEL_DIR}' klasörüne başarıyla kaydedildi.")
    print("--- DBSCAN KÜMELEME TAMAMLANDI ---")


if __name__ == "__main__":
    run_dbscan()