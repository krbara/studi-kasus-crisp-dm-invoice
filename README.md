# Studi Kasus: Analisis & Prediksi Keterlambatan Pembayaran Invoice

Studi kasus berdasarkan kuliah tamu **Informatics Practitioner Lecture Series 2026**
oleh **Bapak Andria Arisal (BRIN)** tentang penerapan analisis data dan AI di dunia industri.

Kasus: **PT Maju Bersama (fiktif)**, pendapatannya naik, tetapi laba bersih turun dan piutang
meningkat. Manajemen ingin tahu apa yang terjadi, mengapa, siapa yang berpotensi telat bayar,
dan tindakan apa yang harus diambil.

> Data yang dipakai adalah **data sintetis** buatan sendiri, dibuat sengaja "berantakan"
> (nilai hilang, duplikat, format tidak konsisten, outlier) untuk menggambarkan kondisi data di lapangan.

## Kerangka CRISP-DM

| Tahap | Implementasi |
|---|---|
| 1. Business Understanding | Menekan piutang macet dengan memprediksi invoice yang berpotensi terlambat dibayar |
| 2. Data Understanding | `generate_raw_data()`, `explore()`: cek ukuran, missing value, duplikat, variasi teks |
| 3. Data Preparation | `clean_data()`, `build_features()`: dedup, standarisasi teks, parsing "Rp", penanganan outlier, imputasi, fitur riwayat pelanggan |
| 4. Modeling | Logistic Regression vs Random Forest |
| 5. Evaluation | ROC-AUC, precision, recall, F1, confusion matrix (split berdasarkan waktu) |
| 6. Deployment (simulasi) | Daftar rekomendasi tindakan untuk invoice yang belum dibayar |

## Empat Level Analitik

| Level | Pertanyaan | Output |
|---|---|---|
| Deskriptif | Apa yang terjadi? | `1_deskriptif.png`, `1_deskriptif_per_kuartal.csv` |
| Diagnostik | Mengapa terjadi? | `2_diagnostik.png` (keterlambatan per segmen, termin, nominal, wilayah) |
| Prediktif | Apa yang akan terjadi? | `3_evaluasi_model.csv`, `3_feature_importance.png` |
| Preskriptif | Apa yang harus dilakukan? | `4_rekomendasi_tindakan.csv` |

## Cara Menjalankan

```bash
pip install -r requirements.txt
python analisis_invoice.py
```


link index.html
https://krbara.github.io/studi-kasus-crisp-dm-invoice/

## Catatan Penting

- **Menghindari data leakage**: fitur riwayat keterlambatan pelanggan (`prev_late_rate`) hanya memakai
  invoice *sebelum* invoice yang sedang diprediksi. Data juga dipisah berdasarkan waktu, bukan acak.
- **Performa model moderat (ROC-AUC sekitar 0,65)**. Ini wajar karena keterlambatan bergantung pada faktor
  yang tidak ada di data (kondisi kas pelanggan, dsb.). Model dipakai sebagai alat bantu prioritas, bukan
  pengambil keputusan.
- **Peran manusia**: sesuai pesan narasumber, hasil model dan rekomendasi tetap harus diverifikasi secara
  kritis oleh manusia. Ambang risiko pada tahap preskriptif adalah asumsi yang perlu disesuaikan dengan
  kebijakan perusahaan.
- Pengembangan lanjutan: coba KNIME untuk membangun *workflow* yang sama dan bandingkan hasilnya,
  tambah model (Gradient Boosting), atau lakukan *hyperparameter tuning*.

## Referensi

Kuliah tamu *Informatics Practitioner Lecture Series 2026*, Andria Arisal, BRIN.
