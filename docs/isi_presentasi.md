# ISI PRESENTASI — SPK Kelayakan Kredit Mikro
# Tugas Besar Desain dan Komputasi Algoritma (DKA) — Semester 4
# Topik: Penilaian Kelayakan Kredit Mikro Menggunakan Logika Fuzzy (Mamdani & Sugeno) + ML + DL

---

## SLIDE 1 — COVER / JUDUL

**Judul Utama:**
> Sistem Pendukung Keputusan (SPK) Kelayakan Kredit Mikro
> Berbasis Logika Fuzzy Mamdani & Sugeno
> Terintegrasi Machine Learning & Deep Learning

**Sub-judul:** Tugas Besar Desain dan Komputasi Algoritma — Semester 4

**Isi konten:**
- Mata Kuliah: Desain dan Komputasi Algoritma (DKA)
- Implementasi: Python + NumPy (from scratch) + Streamlit
- Metode Utama: Fuzzy Mamdani (Centroid) & Sugeno (Weighted Average)
- Model Pendukung: Regresi Linear + ANN/MLP
- Dataset: train.csv — 60.000 sampel bersih

---

## SLIDE 2 — LATAR BELAKANG & MOTIVASI

**Judul:** Mengapa Kredit Mikro Butuh SPK?

**Poin-poin:**
- Penilaian kredit mikro oleh manusia bersifat **subjektif** → rentan bias
- Data nasabah bersifat **fuzzy/ambigu** (tidak ada batas tegas antara "pendapatan cukup" dan "kurang")
- Lembaga keuangan butuh keputusan yang **cepat, konsisten, dan terukur**
- Logika Fuzzy mampu menangani **ketidakpastian linguistik** secara matematis
- Integrasi ML & DL memberikan **validasi tambahan** berbasis pola historis data

**Masalah yang diselesaikan:**
- Bagaimana membangun SPK kredit mikro yang objektif dan berbasis data?
- Metode mana yang lebih akurat: Mamdani (Centroid) atau Sugeno (Weighted Average)?
- Bagaimana perbandingan hasilnya dengan model ML & DL?

---

## SLIDE 3 — VARIABEL INPUT & OUTPUT SISTEM

**Judul:** Variabel Input & Output Fuzzy

**5 Variabel Input (Crisp → Fuzzy):**

| Variabel | Satuan | Himpunan Fuzzy |
|---|---|---|
| Annual Income (Pendapatan) | USD/tahun | Rendah, Sedang, Tinggi |
| Outstanding Debt (Utang) | USD | Sedikit, Sedang, Banyak |
| Interest Rate (Suku Bunga) | % | Rendah, Sedang, Tinggi |
| Delay from Due Date (Keterlambatan) | Hari | Singkat, Sedang, Lama |
| Num of Delayed Payment (Frekuensi Telat) | Kali | Jarang, Sering, SgtSering |

**1 Variabel Output:**
- **Skor Kelayakan Kredit** (0 – 100)
  - BAIK (≥70) → Kredit Disetujui
  - STANDAR (40–70) → Perlu Evaluasi Lanjut
  - BURUK (<40) → Kredit Ditolak

---

## SLIDE 4 — FUNGSI KEANGGOTAAN (MEMBERSHIP FUNCTION)

**Judul:** Fungsi Keanggotaan Input — Trapesium & Segitiga

**Penjelasan:**
- Diimplementasikan **from scratch** menggunakan NumPy (tanpa library fuzzy pihak ketiga)
- Dua jenis MF digunakan:
  - **Trapesium (trapmf):** untuk kategori paling kiri & kanan (nilai ekstrem)
  - **Segitiga (trimf):** untuk kategori tengah (nilai moderat)

**Contoh Parameter MF:**
```
Pendapatan:
  Rendah  = Trapesium(0, 0, 30.000, 50.000)
  Sedang  = Segitiga(30.000, 65.000, 100.000)
  Tinggi  = Trapesium(80.000, 100.000, 1.500.000, 2.000.000)

Suku Bunga:
  Rendah  = Trapesium(0, 0, 8, 12)
  Sedang  = Segitiga(8, 15, 22)
  Tinggi  = Trapesium(18, 25, 35, 100)
```

**[GAMBAR: fitur Kurva Fungsi Keanggotaan Input.jpg]**
*Grafik 2×3 menampilkan kurva MF semua 5 variabel input dengan garis biru putus-putus menunjukkan nilai input saat ini*

---

## SLIDE 5 — PROSES FUZZIFIKASI

**Judul:** Fuzzifikasi — Konversi Nilai Crisp ke Derajat Keanggotaan

**Penjelasan:**
- Setiap nilai input dikonversi ke **derajat keanggotaan (μ)** di masing-masing himpunan fuzzy
- Nilainya berkisar 0.0 (tidak anggota) hingga 1.0 (anggota penuh)
- Nilai dapat berada di **lebih dari satu himpunan** secara bersamaan (overlap)

**Contoh Fuzzifikasi (Input Default):**

| Variabel | Nilai Input | μ Rendah | μ Sedang | μ Tinggi/Lainnya |
|---|---|---|---|---|
| Pendapatan | $45,000 | 0.250 | 0.429 | 0.000 |
| Utang | $1,500 | — | μ Sedikit: 0.500 | μ Sedang: 0.333 |
| Suku Bunga | 12% | 0.000 | 0.571 | 0.000 |
| Keterlambatan | 15 hari | — | μ Singkat: 0.500 | μ Sedang: 0.250 |
| Frek. Telat | 4 kali | — | μ Jarang: 1.000 | μ Sering: 0.000 |

**[GAMBAR: fitur Nilai Fuzzifikasi per Variabel Input.jpg]**
*Panel kartu 5 kolom dengan progress bar menunjukkan derajat keanggotaan tiap variabel*

---

## SLIDE 6 — 34 BASIS ATURAN FUZZY (RULE BASE)

**Judul:** Basis Aturan Fuzzy — 34 Aturan IF-THEN

**Struktur Aturan:**
```
IF [Pendapatan=X] AND [Utang=X] AND [Suku Bunga=X] 
   AND [Keterlambatan=X] AND [Frek.Telat=X]
THEN [Kelayakan=BAIK/STANDAR/BURUK]
```

**Distribusi Aturan:**
- 8 aturan → **BAIK** (kredit layak)
- 11 aturan → **STANDAR** (perlu evaluasi)
- 15 aturan → **BURUK** (kredit ditolak)

**Contoh Aturan Kunci:**
```
R01: inc=Tinggi, debt=Sedikit, ir=Rendah, del=Singkat, num=Jarang → BAIK
R16: inc=Sedang, debt=Sedikit, ir=Sedang, del=Singkat, num=Jarang → BAIK
R11: inc=Rendah, debt=Banyak,  ir=Tinggi, del=Lama,   num=SgtSering → BURUK
R05: inc=Sedang, debt=Sedang,  ir=Sedang, del=Sedang,  num=Sering → STANDAR
```

**Metode Inferensi:** MIN (AND) untuk setiap aturan, MAX untuk agregasi per konsekuensi

**[GAMBAR: fitur Rincian 34 Basis Aturan Fuzzy — Firing Strength per Rule.jpg]**
*Tabel interaktif 34 baris dengan kolom Firing Strength dan status Aktif/Tidak*

---

## SLIDE 7 — INFERENSI & AGREGASI

**Judul:** Proses Inferensi Fuzzy — Firing Strength & Agregasi MAX

**Tahapan Inferensi:**
1. **Fuzzifikasi** → hitung μ setiap variabel input
2. **Evaluasi Aturan** → hitung *firing strength* tiap rule menggunakan operator **MIN**
   - `α_rule = min(μ_inc, μ_debt, μ_ir, μ_del, μ_num)`
3. **Agregasi** → gabungkan semua aturan per konsekuensi menggunakan operator **MAX**
   - `u_baik    = max(firing strength semua aturan BAIK)`
   - `u_standar = max(firing strength semua aturan STANDAR)`
   - `u_buruk   = max(firing strength semua aturan BURUK)`

**Hasil Contoh (Input Default):**
- u_baik = 0.4286
- u_standar = 0.3333
- u_buruk = 0.0000
- Aturan aktif: 3 dari 34 — Dominan: **BAIK**

---

## SLIDE 8 — DEFUZZIFIKASI MAMDANI (CENTROID)

**Judul:** Defuzzifikasi Mamdani — Metode Centroid

**Cara Kerja:**
- Potong setiap MF output dengan firing strength → **daerah aktif**
- Gabungkan semua daerah aktif dengan operator **MAX** → **fungsi agregat**
- Hitung **titik centroid** (pusat massa) area agregat:

```
              ∫ x · μ_agregat(x) dx
z_Mamdani = ─────────────────────────
                ∫ μ_agregat(x) dx
```

**MF Output:**
```
BAIK    = Trapesium(70, 85, 100, 100)
STANDAR = Segitiga(30, 50, 75)
BURUK   = Trapesium(0, 0, 25, 45)
```

**Implementasi:** Integrasi numerik 200 titik (resolusi tinggi)

**Hasil contoh:** Skor Mamdani = **68.69** → Kategori BAIK
**Waktu komputasi:** ~0.922 ms (lebih lambat, lebih presisi)

**[GAMBAR: fitur Visualisasi Output Fuzzy Real-time.jpg]**
*Grafik menampilkan kurva MF output, daerah agregasi (biru), garis centroid Mamdani (merah solid)*

---

## SLIDE 9 — DEFUZZIFIKASI SUGENO (WEIGHTED AVERAGE)

**Judul:** Defuzzifikasi Sugeno — Weighted Average

**Cara Kerja:**
- Tidak menggunakan fungsi keanggotaan output, melainkan **nilai singleton** tetap
- Hitung rata-rata tertimbang berdasarkan firing strength:

```
              u_baik × 90 + u_standar × 50 + u_buruk × 15
z_Sugeno = ────────────────────────────────────────────────
                   u_baik + u_standar + u_buruk
```

**Konstanta Singleton:**
- BAIK = 90
- STANDAR = 50
- BURUK = 15

**Hasil contoh:** Skor Sugeno = **72.50** → Kategori BAIK
**Waktu komputasi:** ~0.005 ms — **181× lebih cepat** dari Mamdani

**Keunggulan:** Ultra-cepat, cocok untuk inference real-time & dataset besar

---

## SLIDE 10 — VISUALISASI KOMPARASI MAMDANI vs SUGENO

**Judul:** Perbandingan Skor Mamdani vs Sugeno

**[GAMBAR: tampilan_awal.jpg]**
*Screenshot tampilan utama aplikasi SPK menampilkan banner "KREDIT DISETUJUI" (hijau) + 4 kartu metrik*

**Ringkasan Komparasi:**

| Metode | Skor | Waktu | Kecepatan |
|---|---|---|---|
| Mamdani (Centroid) | 68.69 | ~0.922 ms | Normal |
| Sugeno (Weighted Avg) | 72.50 | ~0.005 ms | 181× lebih cepat |
| Selisih | 3.81 | — | Konsisten |

**Interpretasi:** Selisih < 5 poin → **Kedua metode konsisten** → Kategori sama: BAIK

---

## SLIDE 11 — ARSITEKTUR APLIKASI

**Judul:** Arsitektur Sistem SPK Kredit Mikro

**Diagram Alur:**
```
[Input Nasabah]
       ↓
[Fuzzifikasi: 5 Variabel × 3 Himpunan]
       ↓
[Evaluasi 34 Aturan — MIN/AND]
       ↓
[Agregasi MAX per Konsekuensi]
       ↓
    ┌──────────────────────────────┐
    ↓                              ↓
[Defuzz. Mamdani]          [Defuzz. Sugeno]
(Centroid — 200 pts)       (Weighted Average)
    ↓                              ↓
[Skor Mamdani]             [Skor Sugeno]
    └──────────────────────────────┘
                   ↓
           [Keputusan Final]
       BAIK / STANDAR / BURUK
               ↓
    [Model Pendukung (Informasi)]
   Regresi Linear + ANN/MLP (DL)
```

**Stack Teknologi:**
- **Backend:** Python 3, NumPy (fuzzy engine from scratch)
- **ML:** Scikit-learn (LinearRegression, MLPClassifier, StandardScaler)
- **Frontend:** Streamlit (web UI interaktif)
- **Data:** train.csv — 60.000 baris setelah cleaning

---

## SLIDE 12 — MODEL PENDUKUNG: ML & DEEP LEARNING

**Judul:** Model Pendukung — Regresi Linear & ANN/MLP

**Catatan Penting:** ML & DL bersifat **informatif saja** — keputusan kredit 100% dari Logika Fuzzy

**[GAMBAR: fitur Model Pendukung — ML & Deep Learning.jpg]**
*Kartu 4 kolom: Regresi Linear (ML) + ANN Prob. Buruk + ANN Prob. Standar + ANN Prob. Baik*

**Regresi Linear (ML):**
- Fitur: Annual Income, Outstanding Debt, Interest Rate, Delay, Num Delayed
- Target: Credit_Score dikodekan numerik (Poor=0, Standard=1, Good=2)
- Dilatih pada 80% × 60.000 sampel bersih
- Hasil: Prediksi = **Standar** (Raw: 0.999)

**ANN/MLP Classifier (Deep Learning):**
- Arsitektur: 5 → 64 → 32 → 16 → 3 (ReLU, Early Stopping)
- Input di-standarisasi dengan StandardScaler
- Output probabilitas per kelas:
  - Prob. Buruk: 25.7%
  - Prob. Standar: **50.8%**
  - Prob. Baik: 23.5%

---

## SLIDE 13 — EVALUASI PERFORMA BATCH

**Judul:** Evaluasi Akurasi — Ground Truth vs Prediksi Fuzzy

**Metodologi:**
- Sampel: **60.000 data** dari train.csv
- Perbandingan prediksi Mamdani & Sugeno dengan label asli (Poor/Standard/Good)
- Metrik: Akurasi, MAE, MSE, RMSE

**[GAMBAR: fitur Evaluasi Performa Batch — Ground Truth vs Prediksi Fuzzy.jpg]**
*Confusion Matrix berdampingan (Mamdani kiri, Sugeno kanan) + tabel metrik lengkap*

**Hasil Evaluasi (60.000 Sampel):**

| Metrik | Mamdani | Sugeno | Lebih Baik |
|---|---|---|---|
| Akurasi | 58.56% | 58.51% | **Mamdani** |
| MAE | 15.8568 | 15.9353 | **Mamdani** |
| MSE | 626.8125 | 634.0533 | **Mamdani** |
| RMSE | 25.0362 | 25.1804 | **Mamdani** |

**Analisis Confusion Matrix:**
- Standard class: Mamdani 80% benar, Sugeno 78% benar
- Good class: Mamdani 22% benar, Sugeno 24% benar
- Keduanya kesulitan membedakan Poor vs Standard → area improvement

---

## SLIDE 14 — ANALISIS KOMPARATIF MAMDANI vs SUGENO

**Judul:** Mamdani vs Sugeno — Kelebihan & Kekurangan

**[GAMBAR: fitur Analisis Komparatif Mamdani vs Sugeno.jpg]**
*Tampilan dua kolom analisis kelebihan-kekurangan dari dalam aplikasi*

**MAMDANI — Centroid:**
| | Kelebihan | Kekurangan |
|---|---|---|
| ✅ | Output kontinu → presisi tinggi | ❌ Komputasi berat (integrasi numerik iteratif) |
| ✅ | Mudah diinterpretasikan (luas area kurva) | ❌ Waktu eksekusi lebih lama |
| ✅ | Sesuai desain rule linguistik | ❌ Resolusi numerik memengaruhi presisi |
| ✅ | Representasi output alami (MF penuh) | ❌ Tidak efisien untuk real-time/embedded |

**SUGENO — Weighted Average:**
| | Kelebihan | Kekurangan |
|---|---|---|
| ✅ | SANGAT cepat (aritmatika sederhana) | ❌ Output diskret (singleton tetap) |
| ✅ | Efisien untuk dataset besar | ❌ Kurang fleksibel tanpa representasi MF |
| ✅ | Tidak perlu MF output | ❌ Konstanta singleton ditentukan manual |
| ✅ | Cocok untuk sistem kontrol | ❌ Kurang "alami" dibanding Mamdani |

**Kesimpulan:** Keduanya berjalan **paralel** sebagai cross-validation. Mamdani unggul **presisi**, Sugeno unggul **kecepatan**.

---

## SLIDE 15 — DEMO APLIKASI STREAMLIT

**Judul:** Demo Langsung — Aplikasi SPK Kredit Mikro

**[GAMBAR: fitur input profil nasabah.jpg]**
*Screenshot sidebar input profil nasabah: Pendapatan, Utang, Suku Bunga, Keterlambatan, Frekuensi Telat*

**Fitur-Fitur Aplikasi:**
1. 🎛️ **Input Profil Nasabah** — Sidebar interaktif (slider + number input)
2. 🎯 **Keputusan Instan** — Banner warna-warni (Hijau=Disetujui, Kuning=Evaluasi, Merah=Ditolak)
3. 📊 **Perbandingan Mamdani vs Sugeno** — 4 kartu metrik (skor, waktu, selisih, speedup)
4. 🔥 **Firing Strength Aturan** — Progress bar 3 kategori (BAIK/STANDAR/BURUK)
5. 📋 **Tabel 34 Aturan Fuzzy** — Detail setiap aturan dengan status aktif/tidak
6. 📈 **Nilai Fuzzifikasi** — Panel kartu per variabel input
7. 🤖 **Model ML & DL** — Referensi prediksi Regresi Linear & ANN
8. 📉 **Visualisasi Output Fuzzy** — Grafik agregasi & defuzzifikasi real-time
9. 🌀 **Kurva MF Input** — 5 grafik fungsi keanggotaan + posisi nilai input
10. 🧪 **Evaluasi Batch 60K** — Confusion Matrix + Akurasi + MAE/MSE/RMSE
11. 🔍 **Analisis Komparatif** — Perbandingan terstruktur Mamdani vs Sugeno

---

## SLIDE 16 — DATA PREPROCESSING & CLEANING

**Judul:** Preprocessing Data — Pembersihan & Seleksi Fitur

**Dataset:** train.csv (Credit Score Dataset)

**Langkah Preprocessing:**
1. **Bersihkan karakter asing** dari kolom numerik (regex: `[^\d.\-]`)
2. **Clip nilai negatif** → Delay & Num Delayed Payment ≥ 0
3. **Drop NaN** setelah konversi numerik
4. **Mapping label** → Poor=0, Standard=1, Good=2 (hapus yang tidak dikenali)
5. **Drop duplikat**
6. **Filter outlier bisnis:**
   - Annual Income: [0, 1.500.000]
   - Interest Rate: [0, 35]
   - Delay & Num Delayed: ≥ 0
7. **Sampling** → 60.000 baris acak (random_state=42)
8. **Train/Test Split** → 80:20

**Fitur yang digunakan:**
- Annual_Income, Outstanding_Debt, Interest_Rate, Delay_from_due_date, Num_of_Delayed_Payment

---

## SLIDE 17 — KESIMPULAN

**Judul:** Kesimpulan

**Poin-poin Kesimpulan:**

1. ✅ **SPK berhasil dibangun** menggunakan Logika Fuzzy Mamdani & Sugeno from scratch (Python + NumPy)
2. 🎯 **34 basis aturan fuzzy** dirancang untuk mencakup berbagai profil nasabah kredit mikro
3. ⚡ **Sugeno 181× lebih cepat** dari Mamdani, namun akurasi Mamdani **sedikit lebih tinggi** (58.56% vs 58.51%)
4. 🔄 **Selisih skor < 5 poin** di sebagian besar kasus → kedua metode **konsisten** dalam keputusan final
5. 🤖 **Model ML & DL** memberikan validasi tambahan tetapi bukan penentu keputusan utama
6. 🌐 **Aplikasi Streamlit** menyajikan antarmuka interaktif, informatif, dan real-time untuk pengambil keputusan
7. 📊 **Akurasi 58.5–58.6%** menunjukkan ruang pengembangan lebih lanjut (aturan lebih banyak, tuning MF)

**Rekomendasi Pengembangan:**
- Tambah jumlah aturan (coverage lebih luas)
- Tuning parameter MF (a, b, c, d) menggunakan data historis
- Integrasikan lebih banyak fitur (payment history, credit utilization, dll.)
- Eksplorasi metode defuzzifikasi lain (MOM, SOM)

---

## SLIDE 18 — PENUTUP & SESI TANYA JAWAB

**Judul:** Terima Kasih — Sesi Tanya Jawab

**Isi:**
- **Sistem berhasil menjawab:** Bagaimana membangun SPK kredit mikro objektif berbasis fuzzy logic
- **Implementasi:** 100% from scratch, siap digunakan secara interaktif
- **Teknologi:** Python · NumPy · Streamlit · Scikit-learn · Matplotlib

**Pertanyaan yang mungkin muncul:**
- *Mengapa menggunakan 34 aturan? Apakah sudah cukup?*
  → Dirancang berdasarkan kombinasi linguistik dari 5 variabel × 3 himpunan, sudah merepresentasikan skenario kredit utama

- *Mengapa akurasi hanya 58%?*
  → Fuzzy logic berbasis aturan manusia, tidak di-training dari data. Untuk akurasi lebih tinggi, parameter bisa dioptimasi dengan GA atau PSO

- *Apa perbedaan Mamdani dan Sugeno dalam konteks nyata?*
  → Mamdani cocok untuk laporan detail (presisi), Sugeno cocok untuk sistem real-time/mobile (kecepatan)

- *Bagaimana cara menjalankan aplikasinya?*
  → `streamlit run app.py` dari direktori yang sama dengan `train.csv`

---

## CATATAN TEKNIS UNTUK PRESENTER

### Info Dataset
- File: `train.csv` (Credit Score Dataset Kaggle)
- Ukuran: ~31 MB, setelah cleaning → 60.000 sampel
- Label: Poor, Standard, Good

### Cara Menjalankan Aplikasi
```bash
cd "d:\SEMESTER 4\DKA\TUBES DKA\bab 4\TUBES DKAA"
streamlit run app.py
```

### Dependensi
```
streamlit, pandas, numpy, matplotlib, scikit-learn
```

### Daftar Gambar yang Tersedia
1. `tampilan_awal.jpg` — Tampilan utama aplikasi
2. `fitur input profil nasabah.jpg` — Sidebar input nasabah
3. `fitur Kurva Fungsi Keanggotaan Input.jpg` — Grafik MF 5 variabel
4. `fitur Nilai Fuzzifikasi per Variabel Input.jpg` — Panel nilai fuzzy
5. `fitur Rincian 34 Basis Aturan Fuzzy — Firing Strength per Rule.jpg` — Tabel aturan
6. `fitur Visualisasi Output Fuzzy Real-time.jpg` — Grafik output fuzzy
7. `fitur Model Pendukung — ML & Deep Learning.jpg` — Kartu ML/DL
8. `fitur Analisis Komparatif Mamdani vs Sugeno.jpg` — Komparasi kelebihan/kekurangan
9. `fitur Evaluasi Performa Batch — Ground Truth vs Prediksi Fuzzy.jpg` — Confusion matrix & metrik
