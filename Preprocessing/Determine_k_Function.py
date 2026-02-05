import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from warnings import filterwarnings

filterwarnings('ignore')  # KMeans uyarılarını gizler

CLUSTERING_DIR = '../data/preprocessed/X_clustering_pca_customer.pkl'
OUTPUT_DIR = '../Outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def determine_optimal_k(data_path=CLUSTERING_DIR):
    """
    K-Means için optimal küme sayısını (k) Elbow Metodu ve Silhouette Skoru ile belirler.
    """
    print("=" * 80)
    print("MÜŞTERİ SEGMENTASYONU - OPTIMAL KÜME SAYISI BELİRLEME BAŞLATILIYOR")
    print("=" * 80)
    start_global_time = time.time()

    # VERİ YÜKLEME
    try:
        X = pd.read_pickle(data_path)
        print(f"✓ PCA uygulanmış Müşteri Verisi (X) başarıyla yüklendi. Boyut: {X.shape}")
    except FileNotFoundError:
        print(f"✗ HATA: Veri dosyası '{data_path}' yolunda bulunamadı. Lütfen ön işleme adımını kontrol edin.")
        return

    # K-DEĞERLERİ İÇİN ANALİZ
    K_MAX = 15
    k_range = range(2, K_MAX + 1)

    wcss = []  # Within-Cluster Sum of Squares (WCSS - Elbow Metodu)
    silhouette_scores = {}  # Silhouette Skoru
    db_scores = {}  # Davies-Bouldin Skoru (DB Index)

    print(f"\n[INFO] K değerleri {k_range.start} ile {K_MAX} arasında test ediliyor...")
    print("-" * 50)

    for k in k_range:
        start_time = time.time()
        # KMeans modelini oluştur ve eğit
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)

        # 1. WCSS Değeri (inertia)
        wcss.append(kmeans.inertia_)

        # 2. Metrikler
        silhouette_scores[k] = silhouette_score(X, kmeans.labels_)
        db_scores[k] = davies_bouldin_score(X, kmeans.labels_)

        elapsed_time = time.time() - start_time
        print(f"K={k:<2} tamamlandı. WCSS: {kmeans.inertia_:.2f}. Süre: {elapsed_time:.2f}s")

    # SONUÇLARI GÖRSELLEŞTİRME

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Optimal Küme Sayısı ($k$) Belirleme Analizi (Müşteri Segmentasyonu)', fontsize=16)

    # 1. Elbow Metodu Grafiği (WCSS)
    ax[0].plot(k_range, wcss, marker='o', linestyle='--', color='blue')
    ax[0].set_title('Elbow Metodu (WCSS) ')
    ax[0].set_xlabel('Küme Sayısı (K)')
    ax[0].set_ylabel('WCSS (Düşük İyidir)')
    ax[0].grid(True)

    # 2. Silhouette Skoru Grafiği
    k_list = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    ax[1].plot(k_list, scores, marker='o', linestyle='--', color='green')
    ax[1].set_title('Silhouette Skoru (Yüksek İyidir) ')
    ax[1].set_xlabel('Küme Sayısı (K)')
    ax[1].set_ylabel('Silhouette Skoru')
    ax[1].grid(True)

    # 3. Davies-Bouldin (DB) Index Grafiği
    db_list = list(db_scores.keys())
    db_values = list(db_scores.values())
    ax[2].plot(db_list, db_values, marker='o', linestyle='--', color='red')
    ax[2].set_title('Davies-Bouldin (DB) Index (Düşük İyidir) ')
    ax[2].set_xlabel('Küme Sayısı (K)')
    ax[2].set_ylabel('DB Index')
    ax[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Grafiği kaydet
    plot_filename = os.path.join(OUTPUT_DIR, 'optimal_k_analysis_customer.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f"\n[INFO] Analiz grafiği '{plot_filename}' klasörüne kaydedildi.")

    # OPTİMAL K YORUMU İÇİN ÇIKTI

    summary_df = pd.DataFrame({
        'WCSS': wcss,
        'Silhouette': list(silhouette_scores.values()),
        'DB_Index': list(db_scores.values())
    }, index=k_range)

    print("\n" + "=" * 55)
    print("K-Değerleri İçin Segmentasyon Metrikleri Özet Tablosu")
    print("=" * 55)

    print(f"{'K':<5} {'WCSS':>15} {'Silhouette':>15} {'DB_Index':>15}")
    print("-" * 55)

    for k in k_range:
        row = summary_df.loc[k]
        # En iyi Silhouette skorunu veya DB skorunu vurgulayabiliriz
        is_best_silhouette = row['Silhouette'] == summary_df['Silhouette'].max()
        is_best_db = row['DB_Index'] == summary_df['DB_Index'].min()

        mark = ''
        if is_best_silhouette:
            mark += ' (Best Sil.)'
        if is_best_db:
            mark += ' (Best DB)'

        print(
            f"{k:<5} "
            f"{row['WCSS']:>15,.2f} "  # WCSS için virgülle ayrılmış, 2 ondalık basamak
            f"{row['Silhouette']:>15.4f} "
            f"{row['DB_Index']:>15.4f}"
            f"{mark}"
        )

    print("-" * 55)
    end_global_time = time.time()
    print(f"Toplam Süre: {end_global_time - start_global_time:.2f} saniye")
    print("=" * 80)


if __name__ == "__main__":
    determine_optimal_k()