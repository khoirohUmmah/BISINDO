# BISINDO MLP Pipeline

Project ini menyediakan pipeline lengkap untuk membangun pengenal alfabet BISINDO berbasis Multi-Layer Perceptron (MLP) menggunakan landmark tangan dari MediaPipe.

## Struktur Proyek

```
bisindo_mlp/
├── bisindo/
│   ├── __init__.py
│   └── utils.py
├── scripts/
│   ├── data_collect.py
│   ├── extract_features.py
│   ├── train_mlp.py
│   ├── eval.py
│   └── infer_realtime.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── runs/
├── requirements.txt
└── README.md
```

## Persiapan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Alur Cepat

1. Kumpulkan data untuk setiap kelas (misal A, B, C, D, E) sebanyak 300–500 sampel per kelas dengan `python scripts/data_collect.py --label A`.
2. Jalankan ekstraksi fitur gabungan: `python scripts/extract_features.py`.
3. Latih model MLP: `python scripts/train_mlp.py`.
4. Evaluasi performa pada test set: `python scripts/eval.py`.
5. Jalankan inferensi real-time: `python scripts/infer_realtime.py`.

## Penjelasan Skrip

### 1. `scripts/data_collect.py`
Merekam data gesture dari webcam dan menyimpan landmark ter-normalisasi ke `data/raw/<label>.csv`.

```bash
python scripts/data_collect.py --label A --samples 500
```

Opsi tambahan:
- `--out`: path file CSV tujuan (default `data/raw/<label>.csv`).

Tekan `s` untuk menyimpan frame saat 1 tangan terdeteksi, dan `q` untuk keluar.

### 2. `scripts/extract_features.py`
Menggabungkan semua file CSV di `data/raw/` menjadi dataset standar dan melakukan normalisasi fitur, encoding label, serta split stratifikasi.

```bash
python scripts/extract_features.py
```

Output penting:
- `data/processed/dataset.csv`
- `data/processed/train_X.npy`, `data/processed/train_y.npy`, dst.
- `models/scaler.joblib`, `models/label_encoder.joblib`

### 3. `scripts/train_mlp.py`
Melatih model MLP TensorFlow pada data train/val dan menyimpan model terbaik serta grafik metrik ke `runs/`.

```bash
python scripts/train_mlp.py
```

Model tersimpan di `models/mlp.h5` dan konfigurasi inferensi di `models/config.json`.

### 4. `scripts/eval.py`
Menghitung metrik akurasi, precision, recall, F1, laporan per kelas, dan menyimpan confusion matrix.

```bash
python scripts/eval.py
```

### 5. `scripts/infer_realtime.py`
Menjalankan inferensi real-time dari webcam dengan kebijakan penolakan untuk dua tangan atau gestur tidak dikenal.

```bash
python scripts/infer_realtime.py --mirror --show-prob --fps
```

Tambahkan `--tts` untuk mengaktifkan text-to-speech ketika prediksi valid berubah.

## Troubleshooting

- **Kamera tidak terbaca**: Pastikan kamera tidak dipakai aplikasi lain dan cek index kamera dengan `--camera-index` (opsional di skrip data/inferensi).
- **MediaPipe tidak mendeteksi tangan**: Perbaiki pencahayaan dan posisi tangan; pastikan tangan terlihat jelas.
- **Mismatch fitur atau ukuran**: Hapus file di `data/processed/` dan `models/` lalu jalankan kembali `extract_features.py` untuk menyelaraskan scaler/encoder dengan data terbaru.

## Tips Meningkatkan Akurasi

- Tambah variasi pose untuk setiap huruf (sudut kamera, rotasi tangan).
- Rekam pada jarak kamera dan pencahayaan yang berbeda untuk memperkaya dataset.
- Gunakan lebih banyak sampel per kelas dan lakukan augmentasi latar belakang jika diperlukan.

## Logging

Setiap skrip menuliskan log ke terminal dan file di folder `runs/` untuk memudahkan reproduksi eksperimen.

## Reproducibility

Semua skrip menggunakan seed tetap (`42`) untuk NumPy, TensorFlow, dan scikit-learn.
