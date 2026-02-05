import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib


OUTPUT_DIR = '../data/preprocessed'
PLOT_DIR = '../Outputs/preprocessing_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("MÜŞTERİ DAVRANIŞLARI KÜMELEME ÖN İŞLEME - PCA DAHİL")
print("=" * 80)
start_time = time.time()

# 1. VERİ YÜKLEME

print("\n[1/7] Veri Yükleniyor ve İlk İnceleme Yapılıyor...")
FILE_PATH = '../data/shopping_behavior_updated.csv'

try:
    df = pd.read_csv(FILE_PATH, encoding='utf-8')
    if df.shape[0] < 2000:
        print(f"✗ HATA: Veri setinde beklenen minimum 2000 satır yok! ({df.shape[0]} satır)")
    print(f"✓ Veri başarıyla yüklendi. Boyut: {df.shape}")
except FileNotFoundError:
    print(f"✗ HATA: Dosya '{FILE_PATH}' bulunamadı! Lütfen dosya yolunu kontrol edin.")
    exit()

# 2. EKSİK VERİ ANALİZİ VE TEMİZLEME

print("\n[2/7] Eksik Veri Analizi ve Temizleme Yapılıyor...")

# Eksik veri kontrolü
missing_before = df.isnull().sum()
if missing_before.sum() > 0:
    pass  # Bu veri setinde eksik veri yoktur, bu yüzden şimdilik atlanmıştır.

print("✓ Eksik veri temizliği tamamlandı (varsa).")
df_cleaned = df.copy()


# 3. ÖZELLİK SEÇİMİ VE MÜHENDİSLİĞİ

print("\n[3/7] Özellik Seçimi ve Mühendisliği Yapılıyor...")

# 1. Çıkarılması Gereken Sütunlar (Noise/Kimlik)
cols_to_drop = ['Customer ID', 'Item Purchased', 'Location']
df_cleaned.drop(cols_to_drop, axis=1, inplace=True)
print(f"Gereksiz sütunlar çıkarıldı. Yeni sütun sayısı: {df_cleaned.shape[1]}")

# 2. Kategorik Sütun Grupları Tanımlanıyor

# Sayısal Özellikler (Doğrudan ölçeklenecek)
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']

# İkili (Binary) Kategorik Özellikler (Evet/Hayır veya 2 Sınıflı)
binary_features = ['Gender', 'Subscription Status', 'Discount Applied', 'Promo Code Used']

# Sıralı (Ordinal) Kategorik Özellikler
# Boyut: S < M < L < XL
# Sıklık: Annually < Quarterly < Every 3 Months < Monthly < Fortnightly < Bi-Weekly < Weekly
ordinal_features = ['Size', 'Frequency of Purchases']

# Nominal (Nominal) Kategorik Özellikler
nominal_features = ['Category', 'Color', 'Season', 'Shipping Type', 'Payment Method']


# SIRALI KODLAMA İÇİN SIRA TANIMLARI

# OrdinalEncoder'a verilecek sıralamalar tanımlanır
size_order = ['S', 'M', 'L', 'XL']
frequency_order = ['Annually', 'Quarterly', 'Every 3 Months', 'Monthly', 'Fortnightly', 'Bi-Weekly', 'Weekly']

ordinal_categories = [size_order, frequency_order]

# 4. AYKIRI DEĞER (OUTLIER) ANALİZİ VE İŞLEME

print("\n[4/7] Aykırı Değer Analizi Yapılıyor...")

# Bu veri setinde aykırı değerleri doğrudan kaldırmak yerine,
# Robustness için ölçekleyiciye bırakmak (StandardScaler ile) daha güvenlidir. Bu yüzden sadece görselleştirme yaptım.

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Yaş Dağılımı
sns.boxplot(y=df_cleaned['Age'], ax=axes[0], color='skyblue')
axes[0].set_title('Yaş Dağılımı ve Aykırı Değerler')
axes[0].set_ylabel('Yaş')

# 2. Harcama Tutarı Dağılımı
sns.boxplot(y=df_cleaned['Purchase Amount (USD)'], ax=axes[1], color='lightcoral')
axes[1].set_title('Satın Alma Tutarı Dağılımı ve Aykırı Değerler')
axes[1].set_ylabel('Satın Alma Tutarı (USD)')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '2_outlier_analysis_customer.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Grafik kaydedildi: 2_outlier_analysis_customer.png")

# 5. ÖZELLİK TANIMLAMA VE PİPELINE OLUŞTURMA

print("\n[5/7] Özellik Pipeline'ı Oluşturuluyor...")

# 1. Sayısal Pipeline: Eksik değerleri medyan ile doldur ve Standartlaştır.
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Sıralı (Ordinal) Pipeline: Eksik değerleri en sık görülen ile doldur ve Sıralı Kodlama yap.

ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal_encoder', OrdinalEncoder(categories=ordinal_categories))
])

# 3. Nominal/İkili Pipeline: One-Hot Encoding uygulanacak sütunları birleştiriyoruz.
# İkili (Binary) sütunlar da OHE ile işlenebilir (2 sütun üretecektir) veya LabelEncoder (1 sütun).
# Tutarlılık için hepsini OHE ile işledim.
ohe_features = nominal_features + binary_features

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer oluşturuluyor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('ord', ord_pipeline, ordinal_features),
    ('cat', cat_pipeline, ohe_features)
])

# Preprocessing uygula
X = df_cleaned.copy()
X_preprocessed = preprocessor.fit_transform(X)

# One-hot encoding sonrası özellik isimleri
ohe_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(ohe_features)
# Sıralı kodlama sonrası özellik isimleri (aynı kalır)
ord_feature_names = ordinal_features
all_feature_names = numerical_features + ord_feature_names + list(ohe_feature_names)

print(f"\n✓ Preprocessing tamamlandı. Toplam özellik sayısı: {X_preprocessed.shape[1]}")

# 6. PCA UYGULAMASI VE GÖRSELLEŞTİRME

print("\n[6/7] PCA Boyut Azaltma Uygulanıyor...")

# Varyans açıklama analizi için tam PCA
pca_full = PCA()
pca_full.fit(X_preprocessed)

cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

# %90 varyans için gerekli bileşen sayısı
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1

# PCA görselleştirmeleri
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

plt.figure(figsize=(10, 6))
plt.plot(cumsum_variance, marker='o', linestyle='--', color='darkblue', markersize=4)

# %90 Eşiği İşaretle
plt.axhline(y=0.90, color='red', linestyle='-', label='90% Varyans Eşiği')
plt.axvline(x=n_components_90 - 1, color='green', linestyle='--',
            label=f'{n_components_90} Bileşen')

plt.title('PCA: Açıklanan Birikimli Varyans Oranı')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Birikimli Açıklanan Varyans Oranı')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '3_pca_detailed_analysis.png'), dpi=300)


# Basitçe %90 varyans ile PCA uyguluyoruz
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_preprocessed)

print(f"\nNihai PCA Bileşen Sayısı: {pca.n_components_}")
print(f"Açıklanan Toplam Varyans: {pca.explained_variance_ratio_.sum():.4f}")

# Grafiklerin kaydedildiğini varsayalım
plt.close()  # Grafik penceresi kapatılır
print(f"✓ Grafik kaydedildi: 3_pca_detailed_analysis.png")


# 7. VERİYİ KAYDETME

print("\n[7/7] İşlenmiş Veri Kaydediliyor...")

# PCA sonrası DataFrame
pca_feature_names = [f'PC_{i + 1}' for i in range(pca.n_components_)]
X_clustering = pd.DataFrame(X_pca, columns=pca_feature_names)

# Normal preprocessing (PCA olmadan) da kaydet
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_feature_names)

# Kaydetme
X_clustering.to_pickle(os.path.join(OUTPUT_DIR, 'X_clustering_pca_customer.pkl'))
X_preprocessed_df.to_pickle(os.path.join(OUTPUT_DIR, 'X_clustering_customer.pkl'))
joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor_customer.pkl'))
joblib.dump(pca, os.path.join(OUTPUT_DIR, 'pca_model_customer.pkl'))


# SONUÇ RAPORU

end_time = time.time()
elapsed = end_time - start_time

print("\n" + "=" * 80)
print("ÖN İŞLEME TAMAMLANDI")
print("=" * 80)
print(f"Toplam Süre: {elapsed:.2f} saniye")
print(f"Başlangıç Veri Boyutu: {df.shape}")
print(f"Nihai Veri Boyutu (PCA): {X_clustering.shape}")
print("=" * 80)