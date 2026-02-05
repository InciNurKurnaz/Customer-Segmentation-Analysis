import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from warnings import filterwarnings

filterwarnings('ignore')


CLUSTERING_DATA_PATH = '../data/preprocessed/X_clustering_pca_customer.pkl'
OUTPUT_DIR = '../Outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  DBSCAN PARAMETRE ÖN AYARI
# PCA sonrası özellik sayısı (22) baz alınarak min_samples hesaplanıyor.
N_FEATURES = 22
MIN_SAMPLES = 2 * N_FEATURES  # Kural: 2 * D = 44


def find_optimal_eps(data_path=CLUSTERING_DATA_PATH, k=MIN_SAMPLES):
    """
    DBSCAN için optimal eps (yarıçap) değerini K-Distance Grafiği ile belirler.
    k (min_samples) değeri, her noktanın k-en yakın komşusuna olan mesafesini hesaplamak için kullanılır.
    """
    print("=" * 70)
    print(f"--- DBSCAN EPS BELİRLEME (k={k}) ---")
    print("=" * 70)

    # VERİ YÜKLEME
    try:
        X = pd.read_pickle(data_path)
        print(f"✓ Veri başarıyla yüklendi. Boyut: {X.shape}")
    except FileNotFoundError:
        print(f"✗ HATA: Veri dosyası '{data_path}' yolunda bulunamadı.")
        return

    # K-DISTANCE HESAPLAMA
    start_time = time.time()

    # NearestNeighbors modelini oluştur (k+1'i alıyoruz çünkü k=0 noktayı kendisi sayar)
    # k burada min_samples değerine eşittir.
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    # Her noktanın k-en yakın komşusuna olan mesafelerini al
    distances, indices = nn.kneighbors(X)

    # Sadece k. (yani k=44.) komşuya olan mesafeyi al ve sırala
    k_distance = distances[:, k - 1]
    k_distance.sort()

    elapsed_time = time.time() - start_time
    print(f"✓ K-Distance hesaplama tamamlandı. Süre: {elapsed_time:.4f} saniye")

    # GÖRSELLEŞTİRME

    plt.figure(figsize=(12, 6))
    plt.plot(range(0, len(k_distance)), k_distance, marker='.', linestyle='', markersize=2, color='darkblue')

    # Grafiği inceleme talimatları
    plt.title(f'K-Distance Grafiği (k = {k})')
    plt.xlabel('Veri Noktaları (Sıralanmış)')
    plt.ylabel(f'{k}. En Yakın Komşu Mesafesi (eps Değeri)')
    plt.grid(True, alpha=0.5)


    plot_filename = os.path.join(OUTPUT_DIR, 'dbscan_k_distance_graph.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f"✓ Grafik '{plot_filename}' klasörüne kaydedildi. Lütfen inceleyin.")


if __name__ == "__main__":
    find_optimal_eps()