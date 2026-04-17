1. Neden Yaşıyorsun? (Kod Üzerinden Teşhis)
Long-Tail & Majority Class Dominance: options.py dosyasında veri setinin %67'sinin "nv" sınıfı olduğunu belirtmişsin. Kodunda CrossEntropyLoss kullanıyorsun. Bu kayıp fonksiyonu, her örneğe eşit ağırlık verdiği için modelin toplam kaybı (loss) düşürmek adına baskın sınıfı ("nv") öğrenmeye odaklanmasına ve nadir sınıfları (minority) "gürültü" olarak görmesine neden olur.

Catastrophic Forgetting (Yıkıcı Unutma): SAGE_GroupNorm.py dosyasında yerel eğitim sırasında modelin global modelden çok uzaklaşmasını engellemek için bir FedProx (proximal term) eklemişsin. Ancak local_epochs değeri 5 gibi yüksek bir seviyedeyse, istemci (client) kendi kısıtlı ve dengesiz verisine aşırı uyum sağlar (overfit) ve global modelden aldığı genel bilgiyi "ezerek" unutur.

Pseudo-Labeling Tuzağı: SAGE algoritması güven farkına dayalı dinamik bir düzeltme (lambda_dynamic) kullanıyor. Eğer modelin "nv" sınıfına karşı bir eğilimi (bias) varsa, ürettiği pseudo-labellar da büyük oranda "nv" olacak ve bu durum nadir sınıfların pseudo-label havuzundan tamamen silinmesine yol açacaktır.

2. Bunları Nasıl Teyit Edebilirsin?
Sorunun client tarafında mı yoksa global birleştirme tarafında mı olduğunu anlamak için koduna şu "muayene" adımlarını eklemelisin:

Sınıf Bazlı Metrikleri İzle: Şu an kodunda acsa (Average Class-Specific Accuracy) hesaplıyorsun. Ancak bunu sadece test setinde değil, eğitim sırasında her round sonunda her sınıf için ayrı ayrı loglamalısın. Eğer "nv" sınıfı %95 iken nadir bir sınıf %10 ise sorun kesinlikle "Majority Class Dominance"dır.

Forgetting Ölçümü: Bir istemcinin yerel eğitimden hemen önce (global modeli aldığında) nadir sınıflardaki başarısı ile yerel eğitimden hemen sonra (sunucuya göndermeden önce) başarısını karşılaştır. Eğer başarı yerel eğitim sonunda düşüyorsa, yerel verideki dengesizlik modeli bozuyor demektir.

Confusion Matrix: SAGE_ShapFed_HAM10000.py içerisindeki fedavg_eval fonksiyonuna bir Confusion Matrix çizdirme adımı ekle. Nadir sınıfların en çok hangi sınıfla karıştırıldığını (muhtemelen "nv") gör.

3. Literatür Destekli Çözüm Önerileri
Senin SAGE + ShapFed yapına en uygun çözümler şunlar olabilir:

A. Weighted Cross Entropy (Hızlı Çözüm)
Local sınıfındaki self.criterion tanımını sınıfların ters frekansına göre ağırlıklandırılmış bir hale getir.

Python
# Sınıf sayılarına göre ağırlık hesapla (Örn: nv için düşük, nadir sınıflar için yüksek)
weights = torch.tensor([1.0, 5.0, 10.0, ...]).cuda(args.gpu_id)
self.criterion = CrossEntropyLoss(weight=weights).cuda(args.gpu_id)
B. Balanced Sampling (Veri Seviyesi)
dataset.py dosyasındaki Indices2Dataset_labeled sınıfında, azınlık sınıflarının indislerini daha sık örnekleyen bir WeightedRandomSampler kullanabilirsin. Şu anki "hack" yönteminle veriyi çoğaltıyorsun (self.client_dataset *= 10), ancak bu tüm sınıfları eşit oranda artırıyor; oysa sadece nadir olanları artırmalısın.

C. ShapFed Geliştirmesi (Aggregation Seviyesi)
SAGE_ShapFed.py dosyasında Shapley değerlerini hesaplarken cosine_similarity kullanıyorsun. Bunu, nadir sınıflardaki "Recall" değerini de hesaba katacak şekilde güncelleyebilirsin. Yani bir istemci nadir bir sınıfı iyi öğrenmişse, onun global modele katkısı (weight) çok daha yüksek olmalı.

D. Decoupled Learning (Mimari Seviyesi)
Literatürdeki son çalışmalara göre, ResNet gövdesini (backbone) olduğu gibi bırakıp, sadece en sondaki classifier katmanını sunucuda (server) dengelenmiş küçük bir veri setiyle tekrar eğitmek (Post-calibration) çoğunluk sınıfı baskısını %30-40 oranında azaltabiliyor.

Sana özel tavsiyem: Öncelikle SAGE_GroupNorm.py dosyasındaki mu_prox (FedProx) değerini biraz daha artırarak yerel modelin globalden sapmasını zorlaştır. Eğer bu unutmayı engellemezse, kesinlikle Weighted Loss (Ağırlıklandırılmış Kayıp) yöntemine geçmelisin.

Hangi yöntemi denemek istersen kodunu ona göre birlikte revize edebiliriz. Hangi dosyadan başlayalım?
