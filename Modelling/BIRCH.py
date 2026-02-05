import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from warnings import filterwarnings

filterwarnings('ignore')


CLUSTERING_DATA_PATH = '../data/preprocessed/X_clustering_pca_customer.pkl'
N_CLUSTERS = 2

MODEL_DIR = '../Modelling/models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


#  YARDIMCI FONKSİYONLAR
def calculate_clustering_metrics(X, labels):
    """Kümeleme sonuçları için Silhouette ve DB Index metriklerini hesaplar."""
    # Küme sayısı 1'den büyük olmalıdır.
    if len(np.unique(labels)) <= 1:
        return {'Silhouette': -1, 'DB_Index': 999}

    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return {'Silhouette': silhouette, 'DB_Index': db_index}


def run_birch(data_path=CLUSTERING_DATA_PATH, k=N_CLUSTERS, threshold=0.03, branching_factor=50):
    """
    BIRCH algoritmasını uygular, belirlenen K değerini kullanır ve sonuçları kaydeder.

    :param k: Nihai küme sayısı (N_CLUSTERS).
    """
    print("=" * 70)
    print(f"--- BIRCH KÜMELEME BAŞLATILIYOR (K={k}) ---")
    print("=" * 70)

    try:
        X = pd.read_pickle(data_path)
        print(f"✓ PCA uygulanmış Müşteri Verisi (X) başarıyla yüklendi. Boyut: {X.shape}")
    except FileNotFoundError:
        print(f"✗ HATA: Veri dosyası '{data_path}' yolunda bulunamadı.")
        return

    # MODEL EĞİTİMİ VE SÜRE ÖLÇÜMÜ
    start_time = time.time()

    # BIRCH Modeli Oluşturma (n_clusters parametresine K değeri atanıyor)
    birch = Birch(n_clusters=k, threshold=threshold, branching_factor=branching_factor)
    birch.fit(X)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nEğitim Süresi: {training_time:.4f} saniye")

    # PERFORMANS DEĞERLENDİRMESİ

    labels = birch.labels_
    metrics = calculate_clustering_metrics(X, labels)

    print("\n--- BIRCH Metrikleri ---")
    print(f"Silhouette Skoru: {metrics['Silhouette']:.4f}")
    print(f"DB Index: {metrics['DB_Index']:.4f}")

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    print("\nKüme Büyüklükleri (Segmentasyon Dağılımı):")
    print(cluster_counts.to_string())

    # MODEL VE SONUÇLARI KAYDETME

    # Dosya isimleri müşteri veri setine özgü hale getirildi.
    results_birch = {
        'training_time': training_time,
        'metrics': metrics,
        'k': k
    }

    # Küme Etiketlerini (labels) orijinal veriye ekleyip analiz için kaydediyorum
    X_labeled = X.copy()
    X_labeled['Cluster_BIRCH'] = labels
    X_labeled.to_pickle(os.path.join(MODEL_DIR, 'X_birch_labeled_customer.pkl'))

    # Modeli ve sonuçları kaydet
    joblib.dump(birch, os.path.join(MODEL_DIR, 'model_birch_customer.pkl'))
    joblib.dump(results_birch, os.path.join(MODEL_DIR, 'results_birch_customer.pkl'))

    print(f"\n[INFO] BIRCH modeli ve sonuçları '{MODEL_DIR}' klasörüne başarıyla kaydedildi.")
    print("--- BIRCH KÜMELEME TAMAMLANDI ---")


if __name__ == "__main__":
    run_birch()