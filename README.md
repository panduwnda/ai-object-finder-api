# 🎯 AI Object Finder API (COCO-SSD)

Sebuah arsitektur REST API cerdas yang memanfaatkan model Deep Learning **COCO-SSD (TensorFlow.js)** untuk melakukan *Object Detection* (Deteksi Objek). 

Berbeda dengan klasifikasi gambar biasa, sistem ini mampu mengenali **banyak objek sekaligus** di dalam satu gambar, memberikan tingkat probabilitas (akurasi), dan mengembalikan **koordinat presisi (Bounding Box / X, Y, Width, Height)** dari letak masing-masing objek tersebut.

## 📸 Screenshot Aplikasi

<img width="1920" height="1997" alt="screencapture-localhost-3001-2026-03-01-09_47_16" src="https://github.com/user-attachments/assets/df81fe88-340f-471f-84cc-4a13312221de" />


## 🚀 Tech Stack Utama

- **Backend / API:** Node.js, Express.js
- **Machine Learning (AI):** TensorFlow.js (`@tensorflow/tfjs`)
- **Deep Learning Model:** COCO-SSD (Mampu mendeteksi 80 kelas objek berbeda)
- **Image Processing:** Jimp (Konversi matriks piksel RGB)
- **Frontend Interaktif:** HTML5 Canvas API (Untuk menggambar Bounding Box secara visual)

## 📌 Fitur Unggulan (End-to-End System)

1. **Multi-Object Detection:** Mampu mendeteksi manusia, kendaraan, hewan, dan perabotan dalam satu *frame* gambar sekaligus.
2. **REST API Endpoint (`/api/detect`):** Menggunakan metode POST untuk menerima payload *Multipart Form Data* (gambar) dan merespons dengan data JSON berisi daftar objek dan koordinatnya.
3. **Data Pre-processing:** Mengekstrak *TypedArray* (Int32Array) dari file gambar mentah sebelum diubah menjadi *Tensor 3D* untuk keperluan *Inference* AI.
4. **Cloud-Ready Architecture:** Dirancang secara *stateless* tanpa dependensi lokal C++, sehingga 100% siap di-deploy ke ekosistem komputasi awan (AWS EC2, GCP Compute Engine, atau Azure).

## 🛠️ Cara Menjalankan (How to Run)

1. Pastikan Node.js terinstal di komputermu.
2. Clone repository ini:
   ```bash
   git clone [https://github.com/panduwnda/ai-object-finder-api.git](https://github.com/panduwnda/ai-object-finder-api.git)
   ```
3. Masuk ke folder proyek dan install seluruh dependensi:
   ```bash
   cd ai-object-finder-api
   npm install

## ☁️ Deployment Architecture (Cloud-Ready)

Sistem pendeteksi objek ini dirancang dengan arsitektur *stateless* dan siap untuk di-deploy ke ekosistem Cloud (AWS / GCP / Azure). Rencana topologi deployment untuk *production* adalah sebagai berikut:

- **IaaS (Infrastructure as a Service):** AWS EC2 atau GCP Compute Engine (Ubuntu Server 22.04 LTS).
- **Process Manager:** Menggunakan **PM2** untuk menjalankan Node.js di *background* agar server AI tidak mati saat session terminal ditutup.
- **Web Server / Proxy:** Menggunakan **Nginx** sebagai *reverse proxy* untuk mem-forward *traffic* dari port 80 (HTTP) ke port 3001 (Node.js server).
- **Storage:** Memisahkan media penyimpanan gambar (*upload destination*) ke AWS S3 atau Google Cloud Storage agar beban komputasi dan *storage* di server utama (EC2) tetap ringan dan optimal untuk *inference* AI.
   ```
4. Jalankan server (Backend & AI Model):
   ```bash
   node server.js
   ```
5. Buka `http://localhost:3001` di *browser* untuk mencoba antarmuka pengujian visualnya.
