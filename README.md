# DeepLearning-ZatureTeshisi
Derin Öğrenme ile Zatüre Teşhisi

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia bağlantısındaki veriler kullanılmaktadır.
Veri seti Test, Train, Validation klasörlerini içermektedir.
Modelimizi eğitmek için test ve train veri setlerini kullanıyoruz. Daha sonra modelimizi validation veri setini kullanarak doğrulanmaktadır.
Programın eğitilmiş model dosyası oluşturması için chest_xray dizini altında bu verilerin download edilerek kopyalanması gerekmektedir.
Yaklaşık 6000 dosya ve 1.16 GB veri olduğu için buraya eklenmemiştir.

"Zature.py" program çalıştığı zaman derin öğrenme ile veri setinin işlenmesi sonucu "zature.h5" model dosyası oluşacaktır.
Tekrar eğitme ihtiyacı olmadan bu model dosyası ile "Zature_test.py" programı ile verilen bir görüntüden zatüre var mı yok mu teşhisi konulmaktadır
