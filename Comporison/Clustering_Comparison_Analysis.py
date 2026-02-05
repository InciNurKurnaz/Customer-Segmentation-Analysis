import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

filterwarnings('ignore')

MODEL_DIR = '../Modelling/models'
OUTPUT_DIR = '../Outputs/comparison'
DATASET_NAME = 'customer'

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# --- KULLANILAN ALGORİTMALARIN LİSTESİ ---
ALGORITHMS_TO_COMPARE = ['kmeans', 'birch', 'dbscan']



def load_all_results():
    """Tüm kümeleme sonuçlarını yükler"""
    results = {}

    print("=" * 80)
    print("KÜMELEME ALGORİTMALARI KARŞILAŞTIRMASI")
    print("=" * 80)

    for algo in ALGORITHMS_TO_COMPARE:
        file_name = f'results_{algo}_{DATASET_NAME}.pkl'
        file_path = os.path.join(MODEL_DIR, file_name)

        try:
            result = joblib.load(file_path)

            # DBSCAN için Küme Sayısını ve Gürültü Oranını hesaplama
            if algo == 'dbscan':
                n_total = 3900  # Müşteri sayısı
                n_noise = result.get('n_noise', 0)
                result['n_clusters'] = result.get('n_clusters', 0)  # Bulunan küme sayısı
                result['noise_percentage'] = (n_noise / n_total) * 100

            # BIRCH ve KMEANS için N_CLUSTERS = 2
            elif algo in ['kmeans', 'birch']:
                result['n_clusters'] = result.get('k', 2)

            results[algo] = result
            print(f"✓ {algo.upper()} sonuçları yüklendi")

        except FileNotFoundError:
            print(f"✗ {algo.upper()} sonuçları bulunamadı. Dosya yolu: {file_path}")

    return results


def create_comparison_table(results):
    """Karşılaştırma tablosu oluşturur"""
    data = []

    for algo_name, result in results.items():
        metrics = result['metrics']

        # Metrikler hesaplanamadıysa N/A göster
        sil_score = f"{metrics['Silhouette']:.4f}" if metrics['Silhouette'] != -1 else 'N/A'
        db_score = f"{metrics['DB_Index']:.4f}" if metrics['DB_Index'] != 999 else 'N/A'

        row = {
            'Algoritma': algo_name.upper(),
            'Küme Sayısı (K)': result.get('n_clusters', 'N/A'),
            'Eğitim Süresi (s)': f"{result['training_time']:.2f}",
            'Silhouette': sil_score,
            'DB Index': db_score,
        }

        # DBSCAN için ekstra bilgi
        if algo_name == 'dbscan':
            row['Gürültü %'] = f"{result.get('noise_percentage', 0):.1f}%"
        else:
            row['Gürültü %'] = 'N/A'

        data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_comprehensive_comparison(results, comparison_df):
    """Kapsamlı karşılaştırma grafiği"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    algorithms = list(results.keys())
    algo_labels = [a.upper() for a in algorithms]
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    # Verileri hazırla (N/A değerleri 0 veya yüksek bir değere çevirerek)
    silhouette_scores = [results[a]['metrics']['Silhouette'] if results[a]['metrics']['Silhouette'] != -1 else 0 for a
                         in algorithms]

    # Önce geçerli DB skorlarını bulalım
    valid_db_scores = [results[a]['metrics']['DB_Index'] for a in algorithms if
                       results[a]['metrics']['DB_Index'] != 999]

    # Geçerli skorların maksimumunu bul ve DBSCAN hata değeri için geçici bir eşik belirle
    db_score_max_valid = max(valid_db_scores) if valid_db_scores else 5.0
    db_error_placeholder = db_score_max_valid + 1.0

    db_scores = []
    for a in algorithms:
        score = results[a]['metrics']['DB_Index']
        if score != 999:
            db_scores.append(score)
        else:
            # Hesaplama hatası varsa, geçici eşik değerini kullan
            db_scores.append(db_error_placeholder)

    # 1. Silhouette Skorları
    ax1 = axes[0]
    bars_sil = ax1.bar(algo_labels, silhouette_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Silhouette Skoru', fontsize=11, weight='bold')
    ax1.set_title('Silhouette Skoru Karşılaştırması\n(Yüksek = İyi) ', fontsize=12, weight='bold')
    ax1.set_ylim([0, max(silhouette_scores) * 1.2])

    # Değerleri yazdır
    for i, (bar, val) in enumerate(zip(bars_sil, silhouette_scores)):
        ax1.text(bar.get_x() + bar.get_width() / 2., val,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontsize=10, weight='bold')

    # 2. Davies-Bouldin Index
    ax2 = axes[1]
    bars_db = ax2.bar(algo_labels, db_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=11, weight='bold')
    ax2.set_title('DB Index Karşılaştırması\n(Düşük = İyi) ', fontsize=12, weight='bold')

    # En iyi (minimum) değeri vurgula
    best_db_idx = np.argmin([s if s != db_error_placeholder else np.inf for s in db_scores])
    bars_db[best_db_idx].set_edgecolor('gold')
    bars_db[best_db_idx].set_linewidth(4)

    for i, (bar, val) in enumerate(zip(bars_db, db_scores)):
        text_val = f'{val:.4f}'
        # Eğer değer hata eşiği ise, N/A yazdır
        if val == db_error_placeholder:
            text_val = 'N/A'

        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 text_val, ha='center', va='bottom',
                 fontsize=10, weight='bold')
    # 3. Eğitim Süreleri
    ax3 = axes[2]
    training_times = [results[a]['training_time'] for a in algorithms]
    ax3.barh(algo_labels, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_xlabel('Eğitim Süresi (saniye)', fontsize=11, weight='bold')
    ax3.set_title('Eğitim Süresi Karşılaştırması', fontsize=12, weight='bold')

    # 4. Küme ve Gürültü Sayıları (Özel grafik)
    ax4 = axes[3]
    cluster_counts = [results[a]['n_clusters'] for a in algorithms]
    noise_percentages = [results[a].get('noise_percentage', 0) for a in algorithms]

    ax4.bar(algo_labels, cluster_counts, color='#3498db', alpha=0.8, label='Küme Sayısı')

    # Gürültü oranını ikinci eksende göster
    ax4_noise = ax4.twinx()
    ax4_noise.plot(algo_labels, noise_percentages, marker='o', color='#e74c3c', label='Gürültü %')
    ax4_noise.set_ylabel('Gürültü Oranı (%)', color='#e74c3c', fontsize=11, weight='bold')
    ax4.set_ylabel('Bulunan Küme Sayısı', fontsize=11, weight='bold')
    ax4.set_title('Küme ve Gürültü Oranı Dağılımı', fontsize=12, weight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4_noise.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'algorithm_comparison_comprehensive.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def main():
    """Ana karşılaştırma fonksiyonu"""
    results = load_all_results()

    if len(results) < 2:  # En az iki algoritma karşılaştırması için
        print("\n✗ Karşılaştırma için yeterli algoritma sonucu bulunamadı!")
        return

    print(f"\n✓ {len(results)} algoritma sonucu yüklendi")

    # Karşılaştırma tablosu oluştur
    print("\n[1/3] Karşılaştırma tablosu oluşturuluyor...")
    comparison_df = create_comparison_table(results)

    # CSV olarak kaydet
    csv_path = os.path.join(OUTPUT_DIR, 'comparison_table_customer.csv')
    comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Tablo kaydedildi: {csv_path}")

    # Konsola yazdır
    print("\n" + "=" * 80)
    print("KARŞILAŞTIRMA TABLOSU")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    # Görselleştirme
    print("\n[2/3] Karşılaştırma grafikleri oluşturuluyor...")
    plot_path = plot_comprehensive_comparison(results, comparison_df)
    print(f"✓ Grafik kaydedildi: {plot_path}")


    print("\n" + "=" * 80)
    print("KARŞILAŞTIRMA ANALİZİ TAMAMLANDI")
    print("=" * 80)
    print(f"\nÇıktı Klasörü: {OUTPUT_DIR}")
    print("Dosyalar:")
    print("  - comparison_table_customer.csv")
    print("  - algorithm_comparison_comprehensive.png")



if __name__ == "__main__":
    main()