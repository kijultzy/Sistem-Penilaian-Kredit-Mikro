# SKRIP PRESENTASI
## SPK Kelayakan Kredit Mikro — Fuzzy Logic Mamdani & Sugeno
### Presenter: Rizky & Emir

---

> **Catatan:**
> - **[R]** = Bagian Rizky yang berbicara
> - **[E]** = Bagian Emir yang berbicara
> - *Teks miring* = instruksi/aksi (klik slide, tunjuk layar, dll.)
> - Durasi estimasi total: **15–20 menit** + tanya jawab

---

## SLIDE 1 — COVER

**[E]**
*[Berdiri, hadap ke audiens, klik slide pertama]*

"Assalamu'alaikum, selamat pagi/siang semuanya.

Perkenalkan, saya Emir, dan di sebelah saya ada Rizky.

Hari ini kami akan mempresentasikan Tugas Besar mata kuliah Desain dan Komputasi Algoritma, dengan judul: **'Sistem Pendukung Keputusan Kelayakan Kredit Mikro Berbasis Logika Fuzzy Mamdani dan Sugeno, Terintegrasi dengan Machine Learning dan Deep Learning.'**

Nanti kami bagi menjadi beberapa bagian — saya akan membawakan bagian latar belakang dan konsep awal, Rizky akan menjelaskan teknis implementasinya, dan kami akan bergantian di beberapa bagian. Langsung saja kita mulai."

---

## SLIDE 2 — LATAR BELAKANG

**[E]**
*[Klik ke slide 2]*

"Pertama kita mulai dari **kenapa** proyek ini dibuat.

Kalau kita bayangkan seorang petugas bank atau koperasi yang menilai apakah seseorang layak dapat kredit mikro, penilaiannya itu seringkali **subjektif**. Dua petugas yang berbeda bisa menghasilkan keputusan berbeda untuk nasabah yang sama. Ini tidak ideal.

Satu hal lagi — data keuangan nasabah itu sifatnya **fuzzy**, atau ambigu. Pendapatan $45.000 per tahun itu cukup atau tidak? Tidak ada batas tegasnya. Makanya kita tidak bisa begitu saja bilang 'ini layak, ini tidak'.

Nah, di sinilah **Logika Fuzzy** hadir — ia mampu menangani ketidakpastian seperti ini secara matematis.

Ada tiga pertanyaan utama yang ingin kami jawab:
Pertama, bagaimana membangun SPK kredit yang objektif dan berbasis data?
Kedua, metode mana yang lebih akurat, Mamdani atau Sugeno?
Dan ketiga, bagaimana hasilnya dibanding model Machine Learning dan Deep Learning?

Untuk menjawab ini, silakan Rizky lanjutkan ke bagian desain sistem."

---

## SLIDE 3 — VARIABEL INPUT & OUTPUT

**[R]**
*[Klik ke slide 3]*

"Terima kasih, Emir. Sekarang kita masuk ke desain sistem.

Sistem kami menerima **5 variabel input** dari data nasabah. *[Tunjuk tabel]* Yaitu: pendapatan tahunan, utang beredar, suku bunga kredit, keterlambatan dalam hari, dan frekuensi berapa kali terlambat bayar.

Masing-masing variabel ini dikategorikan ke dalam **3 himpunan fuzzy** — misalnya pendapatan bisa 'Rendah', 'Sedang', atau 'Tinggi'. Utang bisa 'Sedikit', 'Sedang', atau 'Banyak'. Dan seterusnya.

Dari ke-5 variabel ini, sistem menghasilkan satu **output**: skor kelayakan kredit antara 0 sampai 100.

*[Tunjuk bagian bawah slide]*

Skor ini dikategorikan ke tiga kelas keputusan:
- Skor 70 ke atas → **BAIK**, kredit disetujui
- Skor 40–70 → **STANDAR**, perlu evaluasi lanjut
- Di bawah 40 → **BURUK**, kredit ditolak"

---

## SLIDE 4 — FUNGSI KEANGGOTAAN

**[R]**
*[Klik ke slide 4]*

"Untuk merepresentasikan himpunan fuzzy itu tadi, kami menggunakan dua jenis **fungsi keanggotaan**: Trapesium dan Segitiga. *[Tunjuk gambar grafik di kanan]*

Fungsi **Trapesium** dipakai untuk himpunan di ujung kiri dan kanan — misalnya 'Rendah' dan 'Tinggi'. Bentuknya datar di tengah, lalu naik atau turun di sisi-sisinya.

Fungsi **Segitiga** untuk himpunan tengah seperti 'Sedang'. Puncaknya tepat di titik b dengan nilai 1.0.

Yang perlu dicatat, ini kami implementasikan **from scratch** pakai NumPy — tanpa library fuzzy pihak ketiga. Jadi benar-benar kode murni Python.

*[Tunjuk contoh kode di kiri]*

Contohnya untuk pendapatan: 'Rendah' adalah Trapesium dari 0 sampai 50.000, 'Sedang' Segitiga dari 30.000 sampai 100.000, dan 'Tinggi' Trapesium mulai 80.000 ke atas."

---

## SLIDE 5 — FUZZIFIKASI

**[R]**
*[Klik ke slide 5]*

"Setelah fungsi keanggotaan terdefinisi, langkah pertama proses fuzzy adalah **fuzzifikasi** — mengubah nilai input konkret menjadi derajat keanggotaan.

Contohnya dengan input default di aplikasi kami: pendapatan $45.000, utang $1.500, suku bunga 12%, keterlambatan 15 hari, dan frekuensi terlambat 4 kali.

*[Tunjuk gambar di kanan]*

Hasilnya bisa kita lihat di sini. Pendapatan $45.000 punya derajat keanggotaan 0.250 di himpunan 'Rendah' dan 0.429 di 'Sedang' — jadi dia masuk ke dua himpunan sekaligus. Ini yang namanya **overlap** dalam logika fuzzy.

Frekuensi terlambat 4 kali menghasilkan μ = 1.0 di himpunan 'Jarang' — anggota penuh.

Nilai-nilai μ inilah yang dipakai untuk evaluasi aturan. Saya serahkan ke Emir untuk bagian aturan fuzzy."

---

## SLIDE 6 — 34 BASIS ATURAN FUZZY

**[E]**
*[Klik ke slide 6]*

"Oke terima kasih Rizky. Sekarang kita masuk ke bagian yang mungkin paling berat di sistem ini — **34 basis aturan fuzzy**.

Setiap aturan berbentuk IF-THEN. *[Tunjuk blok kode]*

Misalnya: 'IF pendapatan Tinggi AND utang Sedikit AND suku bunga Rendah AND keterlambatan Singkat AND frekuensi Jarang THEN kelayakan BAIK.'

Aturan-aturan ini kami rancang berdasarkan logika bisnis kredit. Dari 34 aturan itu:
- 8 aturan berujung pada keputusan BAIK
- 11 aturan ke STANDAR
- dan 15 aturan ke BURUK

*[Tunjuk gambar tabel di kanan]*

Di aplikasi, kita bisa lihat setiap aturan beserta **firing strength**-nya — seberapa kuat aturan itu aktif untuk input yang diberikan. Aturan yang firing strength-nya mendekati 1.0 berarti kondisi nasabah sangat cocok dengan aturan tersebut."

---

## SLIDE 7 — INFERENSI & AGREGASI

**[E]**
*[Klik ke slide 7]*

"Proses inferensinya begini.

*[Tunjuk flow diagram di kiri]*

Setelah fuzzifikasi, kita evaluasi setiap aturan dengan operator **MIN**. Jadi firing strength satu aturan = nilai μ paling kecil dari semua kondisi yang terlibat. Kalau ada satu kondisi yang μ-nya 0, aturan itu langsung tidak aktif.

Setelah semua aturan dievaluasi, kita **agregasi** dengan operator MAX — untuk setiap kategori output kita ambil firing strength tertinggi.

*[Tunjuk progress bar di kanan]*

Hasilnya untuk input default: u_baik = 0.4286, u_standar = 0.3333, u_buruk = 0. Dari sini sudah kelihatan kondisi nasabah ini dominan ke arah **BAIK**.

Ketiga nilai ini dikirim ke proses defuzzifikasi. Ada dua metode — Mamdani dan Sugeno. Rizky yang akan jelaskan."

---

## SLIDE 8 — DEFUZZIFIKASI MAMDANI

**[R]**
*[Klik ke slide 8]*

"Kita ke **defuzzifikasi** — yang tugasnya mengubah nilai fuzzy jadi satu angka konkret, yaitu skor akhir kelayakan kredit.

Metode pertama adalah **Mamdani dengan Centroid**.

Caranya: kita 'potong' setiap fungsi keanggotaan output dengan firing strength masing-masing, gabungkan areanya dengan MAX, lalu hitung **titik centroid**-nya.

*[Tunjuk formula]*

Secara matematis itu integral dari x dikali μ_agregat dibagi dengan integral μ_agregat. Kami implementasikan secara numerik dengan 200 titik sampel.

*[Tunjuk grafik di kanan]*

Di grafik ini, garis biru adalah hasil agregasi, dan garis merah solid adalah titik centroid — yaitu **skor Mamdani = 68.69**.

Metode ini cukup berat, sekitar 0.922 ms per inferensi. Tapi hasilnya kontinu dan presisi."

---

## SLIDE 9 — DEFUZZIFIKASI SUGENO

**[R]**
*[Klik ke slide 9]*

"Metode kedua adalah **Sugeno dengan Weighted Average** — jauh lebih ringan.

Sugeno tidak pakai fungsi keanggotaan output. Dia hanya menggunakan **nilai singleton tetap**: BAIK = 90, STANDAR = 50, BURUK = 15. Lalu hitung rata-rata tertimbang berdasarkan firing strength.

*[Tunjuk formula dan perhitungan step-by-step di kanan]*

Kalau kita hitung manual: u_baik × 90 ditambah u_standar × 50, dibagi totalnya... hasilnya **72.50**.

Dari sisi waktu, Sugeno hanya butuh ~0.005 ms — **181 kali lebih cepat** dari Mamdani!

Kedua hasil — 68.69 dan 72.50 — sama-sama masuk kategori BAIK. Jadi keputusan akhirnya sama. Ini yang kami sebut sistem berjalan **paralel sebagai cross-validation**. Emir, lanjut ke tampilan aplikasi?"

---

## SLIDE 10 — TAMPILAN APLIKASI

**[E]**
*[Klik ke slide 10]*

"Oke, setelah tadi Rizky jelasin proses teknisnya, sekarang kita lihat bagaimana semua itu terlihat di aplikasi yang kami buat.

*[Tunjuk screenshot]*

Ini tampilan utama aplikasi SPK kami. Di sebelah kiri ada sidebar untuk input data nasabah. Di bagian tengah-atas ada **banner keputusan** — kalau hasilnya BAIK, bannernya hijau dengan tulisan 'KREDIT DISETUJUI'.

Di bawahnya ada 4 kartu yang menampilkan skor Mamdani, skor Sugeno, selisih keduanya, dan speedup Sugeno.

Ini kami buat dengan Streamlit, jadi tampilan langsung update secara real-time setiap kali nilai input diubah — tidak perlu klik tombol apapun."

---

## SLIDE 11 — PERBANDINGAN MAMDANI VS SUGENO

**[E]**
*[Klik ke slide 11]*

"Biar lebih jelas, ini perbandingan kedua metode secara langsung.

*[Tunjuk tabel]*

Untuk input yang sama — Mamdani 68.69, Sugeno 72.50. Selisihnya 3.81 poin. Keduanya tetap masuk kategori BAIK.

Dari sisi waktu: Mamdani ~0.922 ms, Sugeno ~0.005 ms. Selisih sangat besar.

Kesimpulannya: kalau butuh **presisi dan interpretabilitas**, Mamdani lebih cocok. Kalau butuh **kecepatan untuk sistem real-time**, Sugeno lebih tepat."

---

## SLIDE 12 — ARSITEKTUR SISTEM

**[R]**
*[Klik ke slide 12]*

"Sekarang kita lihat gambaran arsitektur sistem secara keseluruhan.

*[Ikuti alur flow di kiri dari atas ke bawah]*

Alurnya: **input nasabah** → **fuzzifikasi** → **evaluasi 34 aturan dengan MIN** → **agregasi MAX** → dua proses paralel: **defuzzifikasi Mamdani** dan **Sugeno** → **keputusan final**.

*[Tunjuk bagian stack teknologi di kanan]*

Stack teknologinya: backend fuzzy engine Python + NumPy from scratch, machine learning pakai Scikit-learn, frontend Streamlit, dan datanya train.csv yang sudah melalui cleaning."

---

## SLIDE 13 — MODEL ML & DL

**[R]**
*[Klik ke slide 13]*

"Sekarang soal **model pendukung**. ML dan DL yang kami bangun bersifat **informatif saja** — bukan penentu keputusan. Keputusan kredit 100% dari fuzzy logic.

*[Tunjuk screenshot kartu ML/DL]*

Ada dua model. Pertama, **Regresi Linear** — hasilnya memprediksi 'Standar' dengan raw score 0.999. Kedua, **ANN/MLP Classifier** — arsitekturnya 5-64-32-16-3 dengan ReLU dan Early Stopping. Probabilitasnya: 25.7% Buruk, 50.8% Standar, 23.5% Baik.

Menariknya ML bilang 'Standar', sementara Fuzzy bilang 'Baik'. Ini menunjukkan perbedaan pendekatan — Fuzzy berbasis aturan linguistik, ML berbasis pola statistik. Emir, lanjut ke preprocessing?"

---

## SLIDE 14 — PREPROCESSING DATA

**[E]**
*[Klik ke slide 14]*

"Sebelum data bisa dipakai, perlu preprocessing yang cukup panjang.

*[Tunjuk daftar langkah di kiri]*

Dataset aslinya dari Kaggle, train.csv, sekitar 31 MB. Data aslinya tidak bersih — ada karakter aneh, nilai negatif yang tidak masuk akal, duplikat, dan outlier.

Langkah-langkahnya: bersihkan karakter asing pakai regex, clip nilai negatif jadi nol, drop baris NaN, mapping label Poor-Standard-Good ke 0-1-2, drop duplikat, filter outlier bisnis, sampling 60.000 baris secara acak, lalu split 80:20.

*[Tunjuk kartu hasil di kanan]*

Hasilnya 60.000 baris bersih yang siap dipakai."

---

## SLIDE 15 — EVALUASI PERFORMA BATCH

**[E]**
*[Klik ke slide 15]*

"Sekarang, seberapa akurat sistem fuzzy kami kalau dibandingkan label asli dari dataset?

*[Tunjuk tabel metrik]*

Kami evaluasi di 60.000 sampel. Hasilnya: akurasi Mamdani **58.56%**, Sugeno **58.51%** — sangat dekat, Mamdani sedikit unggul di semua metrik.

*[Tunjuk gambar confusion matrix di kanan]*

Dari confusion matrix — untuk kelas Standard, Mamdani benar 80%, Sugeno 78%. Untuk kelas Good, Sugeno sedikit lebih baik, 24% vs 22%.

Akurasi 58% ini wajar karena fuzzy logic kami berbasis aturan yang kami rancang manual, bukan hasil training. Kalau mau akurasi lebih tinggi, parameter MF-nya bisa dioptimasi dengan Genetic Algorithm."

---

## SLIDE 16 — ANALISIS KOMPARATIF

**[R]**
*[Klik ke slide 16]*

"Terima kasih Emir. Kita simpulkan perbandingan Mamdani dan Sugeno secara menyeluruh.

*[Tunjuk dua kolom card]*

**Mamdani:** Output kontinu, presisi tinggi, mudah divisualisasikan. Kekurangannya komputasi berat dan tidak efisien untuk real-time.

**Sugeno:** Sangat cepat, efisien untuk data besar. Kekurangannya output diskret dan konstanta singleton harus ditentukan manual.

*[Tunjuk card kesimpulan]*

Kesimpulannya: keduanya konsisten dengan selisih selalu di bawah 5 poin, dan cocok dijalankan paralel. Tidak ada yang 'menang mutlak' — tergantung kebutuhan."

---

## SLIDE 17 — KESIMPULAN

**[R]**
*[Klik ke slide 17]*

"Oke, kita ke kesimpulan akhir.

*[Tunjuk poin-poin di kiri]*

Satu, sistem SPK berhasil kami bangun dari nol dengan Python dan NumPy tanpa library fuzzy eksternal.

Dua, 34 basis aturan sudah cukup merepresentasikan berbagai profil nasabah kredit mikro.

Tiga, Sugeno **181 kali lebih cepat** dari Mamdani, tapi akurasi Mamdani sedikit lebih unggul — 58.56% vs 58.51%.

Empat, selisih skor keduanya hampir selalu di bawah 5 poin — keputusan final konsisten.

Lima, model ML dan DL berfungsi sebagai validasi silang, bukan penentu utama.

*[Tunjuk bagian rekomendasi di kanan]*

Untuk pengembangan selanjutnya: tambah aturan, tuning MF dengan GA atau PSO, tambah fitur input, dan eksplorasi metode defuzzifikasi lain seperti MOM atau SOM.

Emir, untuk penutupnya."

---

## SLIDE 18 — PENUTUP & TANYA JAWAB

**[E]**
*[Klik ke slide 18]*

"Baik, itu tadi presentasi lengkap dari kami.

Untuk merangkum: kami berhasil membangun sistem pendukung keputusan kredit mikro berbasis fuzzy logic Mamdani dan Sugeno yang dijalankan paralel, divalidasi dengan Machine Learning dan Deep Learning, dan dikemas dalam antarmuka web interaktif dengan Streamlit.

Kalau ingin mencoba langsung, bisa dijalankan lewat perintah `streamlit run app.py` di direktori project-nya.

Demikian presentasi dari kami — Rizky dan Emir. Kami buka untuk sesi tanya jawab. Silakan kalau ada yang ingin ditanyakan."

---

## PANDUAN MENJAWAB PERTANYAAN

---

**Q: "Mengapa akurasi fuzzy-nya hanya sekitar 58%?"**

**[R atau E]:**
"Akurasi 58% itu memang karakteristik dari fuzzy logic berbasis aturan linguistik. Sistem kami tidak 'belajar' dari data seperti machine learning — aturannya kami rancang secara manual berdasarkan logika bisnis kredit. Kalau ingin akurasi lebih tinggi, parameter fungsi keanggotaannya bisa dioptimasi menggunakan Genetic Algorithm atau Particle Swarm Optimization."

---

**Q: "Kenapa harus 34 aturan? Apa dasarnya?"**

**[E]:**
"Aturan-aturan itu kami rancang dari kombinasi linguistik 5 variabel dengan 3 himpunan masing-masing. Secara teori ada 3⁵ = 243 kombinasi, tapi tidak semua relevan secara bisnis. Kami memilih 34 yang paling representatif — mulai dari yang jelas layak, jelas tidak layak, sampai yang ada di tengah-tengah."

---

**Q: "Apa bedanya Mamdani dan Sugeno dalam praktiknya?"**

**[R]:**
"Mamdani lebih cocok kalau kita perlu laporan yang bisa divisualisasikan dan diinterpretasikan lewat kurva. Sugeno lebih cocok kalau sistemnya perlu cepat, misalnya untuk mobile app atau sistem embedded. Dalam sistem kami, keduanya jalan paralel karena kelebihan masing-masing saling melengkapi."

---

**Q: "Mengapa pakai Streamlit?"**

**[E]:**
"Streamlit dipilih karena sangat cocok untuk prototyping aplikasi data science secara cepat. Dengan beberapa baris kode Python kita langsung dapat antarmuka web yang interaktif tanpa perlu menulis HTML atau JavaScript sendiri."

---

> Tips: Kalau ada pertanyaan yang tidak yakin jawabannya,
> lebih baik jujur bilang "kami akan cek lebih lanjut"
> daripada menjawab dengan tidak yakin.
