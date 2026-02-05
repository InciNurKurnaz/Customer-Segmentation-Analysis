# Customer Segmentation & Shopping Behavior Analysis
Bu proje, müşteri alışveriş alışkanlıklarını analiz ederek veriye dayalı pazarlama stratejileri geliştirmek amacıyla Gözetimsiz Öğrenme (Unsupervised Learning) tekniklerini kullanır. 
Proje; veri ön işleme, boyut indirgeme (PCA) ve gelişmiş kümeleme algoritmalarını içeren tam bir veri bilimi hattına sahiptir.


--- 

## 📊 Veri Seti ve İşleme
Veri seti, müşterilerin yaş, harcama tutarı, abonelik durumu ve alışveriş sıklığı gibi demografik ve davranışsal özelliklerini içerir.

### Veri Ön İşleme (Pipeline)

Özellik Mühendisliği: Gereksiz kimlik bilgileri temizlendi, kategorik veriler için One-Hot Encoding ve sıralı veriler için Ordinal Encoding uygulandı.  
Ölçeklendirme: Mesafe tabanlı algoritmaların başarısı için StandardScaler kullanıldı.  
Boyut İndirgeme (PCA): Verideki varyansın %90'ını temsil eden en önemli bileşenler seçilerek veri boyutu optimize edildi.  

--- 

## 🤖 Kullanılan Kümeleme Algoritmaları
En iyi segmentasyon sonucuna ulaşmak için üç farklı yaklaşım karşılaştırılmıştır:  

K-Means: Mesafe tabanlı, hızlı ve etkili kümeleme.  
BIRCH: Büyük veri setleri için ölçeklenebilir hiyerarşik kümeleme.  
DBSCAN: Yoğunluk tabanlı, gürültüye (aykırı değerlere) dayanıklı kümeleme.  

--- 

## 📈 Model Değerlendirme ve Optimal K Seçimi
Küme sayısını ($k$) belirlemek için çok kriterli bir değerlendirme yapılmıştır:  

Elbow Method (Dirsek Yöntemi): WCSS değerindeki değişim izlendi.  
Silhouette Score: Kümelerin birbirine uzaklığı ve kendi içindeki yoğunluğu ölçüldü.  
Davies-Bouldin Index: Küme içi benzerlik ve kümeler arası fark analiz edildi.
