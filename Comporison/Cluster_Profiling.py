import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

filterwarnings('ignore')


DATA_PATH = '../data/shopping_behavior_updated.csv'
LABELED_DATA_PATH = '../Modelling/models/X_kmeans_labeled_customer.pkl'
ORIGINAL_PREPROCESSOR_PATH = '../data/preprocessed/preprocessor_customer.pkl'
OUTPUT_DIR = '../Outputs/cluster_profiling'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")


def load_and_merge_data(labeled_path, algo_name):
    """Orijinal veriyi ve belirli bir algoritmanın etiketlerini yükleyip birleştirir."""
    try:
        df_original = pd.read_csv(DATA_PATH, encoding='utf-8')
        df_labeled_pca = pd.read_pickle(labeled_path)

        # Etiketleri ekle
        df_original['Cluster'] = df_labeled_pca[f'Cluster{"" if algo_name == "K-Means" else "_" + algo_name}'].values

        print(f"✓ {algo_name.upper()} Etiketleri Yüklendi ve Birleştirildi.")

        # DBSCAN için Gürültüyü (-1) Hariç Tut
        if algo_name == 'DBSCAN':
            df_clustered = df_original[df_original['Cluster'] != -1].copy()
            print(f"  > DBSCAN İçin Gürültü (-1 Etiketli) Hariç Tutuldu. Kalan Kayıt: {df_clustered.shape[0]}")
        else:
            df_clustered = df_original.copy()

        return df_clustered

    except FileNotFoundError as e:
        print(f"✗ HATA: Gerekli dosya bulunamadı: {e}")
        return None


def assign_cluster_names(df, algo_name):
    """Kümeleri analiz sonucuna göre adlandırır ve DataFrame'e ekler."""

    # K-MEANS ve BIRCH (Genel Segmentasyon)
    if algo_name in ['K-Means', 'BIRCH']:

        label_map = {
            0: "Ortalama Sıklıkta Alıcı",
            1: "Daha Uzun Döngülü Tüketici"
        }
        df['Cluster_Name'] = df['Cluster'].map(label_map)

    # DBSCAN (Niş Segmentasyon)
    elif algo_name == 'DBSCAN':

        dbscan_map = {
            0: "Niş Segment A",
            1: "Niş Segment B"
        }
        df['Cluster_Name'] = df['Cluster'].map(dbscan_map)

    return df


def calculate_cluster_profiles(df):
    """Her küme için sayısal ve kategorik ortalamaları hesaplar."""

    # 1. Sayısal Özellikler için Ortalama Hesaplama
    numerical_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    profile_summary = df.groupby('Cluster_Name')[numerical_cols].mean().T

    # 2. Kategorik Özellikler için En Sık Görülen Değerler
    categorical_cols = ['Gender', 'Category', 'Color', 'Season', 'Shipping Type', 'Payment Method',
                        'Frequency of Purchases']

    categorical_profiles = {}

    for col in categorical_cols:
        mode_by_cluster = df.groupby('Cluster_Name')[col].agg(lambda x: x.mode()[0])

        percentage_by_cluster = df.groupby('Cluster_Name')[col].value_counts(normalize=True).mul(100).rename(
            'Percentage').reset_index()

        profile_row = {}
        for name in df['Cluster_Name'].unique():
            most_common_value = mode_by_cluster.loc[name]
            percentage = percentage_by_cluster[
                (percentage_by_cluster['Cluster_Name'] == name) &
                (percentage_by_cluster[col] == most_common_value)
                ]['Percentage'].iloc[0]

            profile_row[name] = f"{most_common_value} ({percentage:.1f}%)"

        categorical_profiles[col] = profile_row

    df_cat_profile = pd.DataFrame(categorical_profiles).T
    df_cat_profile.columns = df['Cluster_Name'].unique()  # Sütun isimleri güncellendi

    return profile_summary, df_cat_profile


def plot_profiling_graphs(df, profile_summary, algo_name):
    """Küme profillerini görselleştirir."""


    # 1. Sayısal Ortalamalar Grafiği
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    numeric_features = profile_summary.index
    plot_data = profile_summary

    plot_data.T.plot(kind='bar', ax=axes[0], rot=0, alpha=0.8, legend=False)
    axes[0].set_title(f'Küme Ortalamaları Karşılaştırması ({algo_name})', fontsize=14, weight='bold')
    axes[0].set_ylabel('Ortalama Değer (Standardize Edilmemiş)', fontsize=10)
    axes[0].legend(title='Özellik', fontsize=8, loc='upper left')
    axes[0].grid(axis='y', alpha=0.5)

    # 2. Satın Alma Tutarı Dağılımı (Kutu Grafiği)
    sns.boxplot(x='Cluster_Name', y='Purchase Amount (USD)', data=df, ax=axes[1])
    axes[1].set_title(f'Satın Alma Tutarı Dağılımı ({algo_name})', fontsize=14, weight='bold')
    axes[1].set_xlabel('Küme Adı')
    axes[1].set_ylabel('Satın Alma Tutarı (USD)')
    axes[1].tick_params(axis='x', rotation=15)

    # 3. Satın Alma Sıklığı Dağılımı (Bar Grafiği)
    sns.countplot(x='Frequency of Purchases', hue='Cluster_Name', data=df, ax=axes[2])
    axes[2].set_title(f'Satın Alma Sıklığına Göre Küme Dağılımı ({algo_name})', fontsize=14, weight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
    axes[2].set_xlabel('Satın Alma Sıklığı')
    axes[2].set_ylabel('Müşteri Sayısı')
    axes[2].legend(title='Küme Adı')

    # 4. Kategoriye Göre Dağılım (Bar Grafiği)
    sns.countplot(x='Category', hue='Cluster_Name', data=df, ax=axes[3])
    axes[3].set_title(f'Kategoriye Göre Küme Dağılımı ({algo_name})', fontsize=14, weight='bold')
    axes[3].set_xlabel('Kategori')
    axes[3].set_ylabel('Müşteri Sayısı')
    axes[3].legend(title='Küme Adı')

    plt.suptitle(f'{algo_name} Müşteri Segment Profilleri', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f'cluster_profiles_{algo_name.lower()}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Profilleme grafikleri kaydedildi: {plot_path}")


def main_profiling_all_algos():
    """Tüm algoritmaların profilleme iş akışını yönetir."""

    profiling_configs = {
        'K-Means': '../Modelling/models/X_kmeans_labeled_customer.pkl',
        'BIRCH': '../Modelling/models/X_birch_labeled_customer.pkl',
        'DBSCAN': '../Modelling/models/X_dbscan_labeled_customer.pkl'
    }

    for algo_name, labeled_path in profiling_configs.items():
        print("\n" + "=" * 80)
        print(f"MÜŞTERİ KÜME PROFİLLEMESİ BAŞLATILIYOR ({algo_name})")
        print("=" * 80)

        df_clustered = load_and_merge_data(labeled_path, algo_name)

        if df_clustered is not None and not df_clustered.empty:
            df_clustered_named = assign_cluster_names(df_clustered, algo_name)
            profile_summary, df_cat_profile = calculate_cluster_profiles(df_clustered_named)
            plot_profiling_graphs(df_clustered_named, profile_summary, algo_name)

            # Sonuç tablolarını kaydet
            profile_summary.to_csv(os.path.join(OUTPUT_DIR, f'numerical_profiles_{algo_name.lower()}.csv'))
            df_cat_profile.to_csv(os.path.join(OUTPUT_DIR, f'categorical_profiles_{algo_name.lower()}.csv'))

            print(f"✓ {algo_name} Profilleme tabloları kaydedildi.")
        else:
            print(f"✗ {algo_name} için geçerli veri bulunamadı (Tüm veriler gürültü olabilir). Profilleme atlandı.")

    print("\n" + "=" * 80)
    print("TÜM PROFİLLEMELER TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main_profiling_all_algos()