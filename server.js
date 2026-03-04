const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");

// Menggunakan Pure JS untuk menghindari error C++
const tf = require("@tensorflow/tfjs");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const Jimp = require("jimp");

const app = express();
const PORT = 3001; 

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage: storage });

let aiModel;
async function loadAIModel() {
  console.log("⏳ Sedang memuat Model AI (COCO-SSD Object Detection)...");
  try {
    // Memuat model COCO-SSD (mampu mendeteksi 80 jenis objek berbeda)
    aiModel = await cocoSsd.load();
    console.log("✅ Model AI COCO-SSD berhasil dimuat! Server siap.");
  } catch (error) {
    console.error("❌ Gagal memuat model:", error);
  }
}
loadAIModel();

// REST API ENDPOINT: POST /api/detect
app.post("/api/detect", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ status: "error", message: "Tidak ada gambar." });
    if (!aiModel) return res.status(503).json({ status: "error", message: "Model AI masih dimuat." });

    console.log(`Menganalisis objek pada gambar: ${req.file.originalname}...`);

    // 1. IMAGE PROCESSING (Membaca matriks piksel)
    const image = await Jimp.read(req.file.path);

    // Kita tidak meresize gambar di sini agar koordinat kotaknya akurat dengan gambar asli
    const width = image.bitmap.width;
    const height = image.bitmap.height;
    const numPixels = width * height;
    const values = new Int32Array(numPixels * 3);
    const p = image.bitmap.data;

    for (let i = 0; i < numPixels; i++) {
      values[i * 3 + 0] = p[i * 4 + 0]; // R
      values[i * 3 + 1] = p[i * 4 + 1]; // G
      values[i * 3 + 2] = p[i * 4 + 2]; // B
    }

    const tensor = tf.tensor3d(values, [height, width, 3], "int32");

    // 2. AI MENDETEKSI OBJEK (Bounding Boxes)
    const predictions = await aiModel.detect(tensor);

    tensor.dispose();
    fs.unlinkSync(req.file.path); // Bersihkan file

    // 3. KEMBALIKAN KOORDINAT KE CLIENT
    res.json({
      status: "success",
      message: `Berhasil mendeteksi ${predictions.length} objek`,
      data: predictions,
    });
  } catch (error) {
    console.error("Error:", error);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ status: "error", message: error.message });
  }
});

// UI FRONTEND (Single Page Application)
app.get("/", (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="id" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Object Finder - COCO SSD</title>
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <!-- Tailwind CSS -->
        <script src="https://cdn.tailwindcss.com"></script>
        <!-- Phosphor Icons -->
        <script src="https://unpkg.com/@phosphor-icons/web"></script>
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        fontFamily: { 
                            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
                            mono: ['"JetBrains Mono"', 'monospace']
                        },
                        colors: {
                            brand: { 50: '#ecfeff', 100: '#cffafe', 400: '#22d3ee', 500: '#06b6d4', 600: '#0891b2', 900: '#164e63' }
                        }
                    }
                }
            }
        </script>
        <style>
            body { background-color: #0f172a; background-image: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%); }
            .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); }
            .hide { display: none !important; }
            .drag-active { border-color: #22d3ee !important; background-color: rgba(34, 211, 238, 0.05) !important; }
            canvas { max-width: 100%; height: auto; border-radius: 0.75rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); }
        </style>
    </head>
    <body class="min-h-screen text-slate-200 flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">
        
        <!-- Header -->
        <div class="text-center mb-10 w-full max-w-2xl">
            <div class="inline-flex items-center justify-center p-3 bg-brand-500/20 border border-brand-500/30 rounded-2xl mb-4 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                <i class="ph ph-scan text-4xl text-brand-400"></i>
            </div>
            <h1 class="text-3xl md:text-5xl font-extrabold tracking-tight text-white mb-3">
                Deep<span class="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-blue-500">Scan</span> AI
            </h1>
            <p class="text-slate-400 text-sm md:text-base">Unggah gambar dengan banyak objek di dalamnya. AI kami akan menemukan dan menandai koordinatnya secara akurat.</p>
        </div>

        <!-- Main Card -->
        <div class="w-full max-w-3xl glass-card rounded-3xl shadow-2xl p-6 md:p-8 relative">
            
            <form id="uploadForm" class="space-y-6">
                <!-- Dropzone & Canvas Area -->
                <div id="dropzone" class="relative group border-2 border-dashed border-slate-700 rounded-2xl p-2 md:p-8 text-center hover:bg-slate-800/50 hover:border-brand-500/50 transition-all duration-300 cursor-pointer min-h-[300px] flex flex-col justify-center items-center overflow-hidden">
                    <input type="file" id="imageInput" accept="image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" required>
                    
                    <!-- Placeholder UI -->
                    <div id="uploadPlaceholder" class="flex flex-col items-center justify-center space-y-4 pointer-events-none p-6">
                        <div class="p-4 bg-slate-800 rounded-full group-hover:scale-110 group-hover:bg-brand-500/20 group-hover:text-brand-400 transition-all duration-300 text-slate-400">
                            <i class="ph ph-image-square text-5xl"></i>
                        </div>
                        <div>
                            <p class="text-base font-semibold text-slate-200">Klik atau seret gambar ke sini</p>
                            <p class="text-xs text-slate-500 mt-1">AI mendeteksi hingga 80 kategori objek (COCO-SSD)</p>
                        </div>
                    </div>

                    <!-- Canvas untuk menggambar gambar dan hasil deteksi -->
                    <div id="canvasWrapper" class="hide w-full relative z-20 flex flex-col items-center justify-center">
                        <canvas id="outputCanvas" class="bg-slate-900 border border-slate-700"></canvas>
                        <p id="fileName" class="text-xs font-mono text-slate-400 mt-3 bg-slate-800/80 px-3 py-1 rounded-full"></p>
                    </div>
                </div>

                <!-- Control Button -->
                <div class="flex flex-col sm:flex-row gap-4 pt-2">
                    <button type="submit" id="submitBtn" class="flex-1 relative inline-flex items-center justify-center px-8 py-3.5 text-base font-bold text-slate-900 transition-all duration-200 bg-brand-400 border border-transparent rounded-xl hover:bg-brand-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-brand-400 shadow-[0_0_20px_rgba(34,211,238,0.4)] hover:shadow-[0_0_25px_rgba(34,211,238,0.6)] disabled:opacity-70 disabled:cursor-not-allowed">
                        <i class="ph ph-crosshair text-xl mr-2"></i>
                        <span>Mulai Pemindaian Objek</span>
                    </button>
                    <button type="button" id="resetBtn" class="hide inline-flex items-center justify-center px-6 py-3.5 text-base font-semibold text-slate-300 transition-all duration-200 bg-slate-800 border border-slate-700 rounded-xl hover:bg-slate-700 focus:outline-none">
                        <i class="ph ph-trash text-lg mr-2"></i> Reset
                    </button>
                </div>
            </form>

            <!-- Status & Loading -->
            <div id="statusContainer" class="hide mt-6 bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <i id="statusIcon" class="ph ph-spinner-gap text-2xl text-brand-400 animate-spin"></i>
                    <div>
                        <p id="statusText" class="text-sm font-semibold text-slate-200">Menganalisis matriks gambar...</p>
                        <p id="statusSubtext" class="text-xs text-slate-400 mt-0.5">Memproses Tensor melalui Neural Network</p>
                    </div>
                </div>
            </div>

            <!-- Results Section (Badges & JSON) -->
            <div id="resultSection" class="hide mt-8 border-t border-slate-700 pt-6">
                
                <h3 class="text-sm font-semibold text-slate-300 flex items-center mb-3 uppercase tracking-wider">
                    <i class="ph ph-bounding-box text-brand-400 mr-2 text-lg"></i> Objek Terdeteksi
                </h3>
                
                <!-- Container untuk daftar objek (Pills/Badges) -->
                <div id="detectedBadges" class="flex flex-wrap gap-2 mb-6">
                    <!-- Badges injected by JS -->
                </div>

                <!-- Raw JSON Toggle -->
                <details class="group border border-slate-700/60 rounded-xl overflow-hidden bg-slate-800/30">
                    <summary class="text-xs text-slate-400 cursor-pointer hover:bg-slate-800/80 hover:text-slate-200 p-3 font-medium list-none flex items-center justify-between transition-colors">
                        <span class="flex items-center"><i class="ph ph-braces mr-2 text-lg text-brand-500"></i> Lihat Detail Koordinat (JSON API)</span>
                        <i class="ph ph-caret-down transition-transform group-open:-rotate-180"></i>
                    </summary>
                    <div class="bg-[#0d1117] relative border-t border-slate-700/60">
                        <div class="absolute top-2 right-2 text-brand-500/50 text-[10px] font-mono">response.data</div>
                        <pre id="jsonResult" class="p-4 text-emerald-400 font-mono text-xs sm:text-sm overflow-x-auto max-h-64 overflow-y-auto"></pre>
                    </div>
                </details>
            </div>

        </div>

        <script>
            // Elemen DOM
            const imageInput = document.getElementById('imageInput');
            const dropzone = document.getElementById('dropzone');
            const uploadPlaceholder = document.getElementById('uploadPlaceholder');
            const canvasWrapper = document.getElementById('canvasWrapper');
            const outputCanvas = document.getElementById('outputCanvas');
            const ctx = outputCanvas.getContext('2d');
            const fileName = document.getElementById('fileName');
            const form = document.getElementById('uploadForm');
            const submitBtn = document.getElementById('submitBtn');
            const resetBtn = document.getElementById('resetBtn');
            
            const statusContainer = document.getElementById('statusContainer');
            const statusIcon = document.getElementById('statusIcon');
            const statusText = document.getElementById('statusText');
            const statusSubtext = document.getElementById('statusSubtext');
            
            const resultSection = document.getElementById('resultSection');
            const detectedBadges = document.getElementById('detectedBadges');
            const jsonResult = document.getElementById('jsonResult');

            let currentImage = null; // Menyimpan gambar asli untuk redraw

            // Handle Drag & Drop Visuals
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
            });
            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, () => dropzone.classList.add('drag-active'), false);
            });
            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, () => dropzone.classList.remove('drag-active'), false);
            });

            // 1. Fungsi saat user memilih gambar
            imageInput.addEventListener('change', async function() {
                const file = this.files[0];
                if (!file) return;

                // Reset UI state
                resultSection.classList.add('hide');
                statusContainer.classList.add('hide');
                resetBtn.classList.remove('hide');
                
                fileName.innerText = file.name;
                uploadPlaceholder.classList.add('hide');
                canvasWrapper.classList.remove('hide');

                // Load image to canvas
                currentImage = new Image();
                currentImage.src = URL.createObjectURL(file);
                
                await new Promise(resolve => currentImage.onload = resolve);
                
                // Sesuaikan ukuran canvas dengan gambar asli
                outputCanvas.width = currentImage.width;
                outputCanvas.height = currentImage.height;
                
                // Gambar original image
                ctx.drawImage(currentImage, 0, 0);
            });

            // Fungsi Reset
            resetBtn.addEventListener('click', () => {
                imageInput.value = '';
                currentImage = null;
                ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                uploadPlaceholder.classList.remove('hide');
                canvasWrapper.classList.add('hide');
                resultSection.classList.add('hide');
                statusContainer.classList.add('hide');
                resetBtn.classList.add('hide');
            });

            // 2. Submit API Call
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                if(!imageInput.files.length || !currentImage) return;

                // Pastikan gambar di-reset tanpa kotak (jika di-submit ulang)
                ctx.drawImage(currentImage, 0, 0);

                // UI Loading State
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="ph ph-spinner-gap animate-spin mr-2 text-xl"></i><span>Memindai...</span>';
                
                statusContainer.classList.remove('hide');
                statusIcon.className = "ph ph-spinner-gap text-2xl text-brand-400 animate-spin";
                statusText.innerText = "Menganalisis matriks gambar...";
                statusSubtext.innerText = "Deep Learning sedang memproses Tensor...";
                statusText.classList.remove('text-red-400');
                
                resultSection.classList.add('hide');
                detectedBadges.innerHTML = ''; // Bersihkan badge lama

                const formData = new FormData(); 
                formData.append('image', imageInput.files[0]);

                try {
                    const response = await fetch('/api/detect', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if(data.status !== 'success') throw new Error(data.message || "Gagal memproses");

                    // Tampilkan Data JSON Raw
                    jsonResult.innerText = JSON.stringify(data, null, 2);
                    
                    if(data.data.length === 0) {
                        statusIcon.className = "ph ph-info text-2xl text-blue-400";
                        statusText.innerText = "Tidak ada objek yang dikenali.";
                        statusSubtext.innerText = "AI tidak dapat menemukan objek yang masuk dalam kategorinya.";
                    } else {
                        statusIcon.className = "ph ph-check-circle text-2xl text-emerald-400";
                        statusText.innerText = \`Pencarian Selesai! Ditemukan \${data.data.length} objek.\`;
                        statusSubtext.innerText = "Koordinat bounding box berhasil dipetakan.";
                        
                        // Gambar Kotak dan Buat Badges
                        drawBoundingBoxes(data.data);
                        renderBadges(data.data);
                        resultSection.classList.remove('hide');
                    }

                } catch (err) { 
                    statusIcon.className = "ph ph-warning-circle text-2xl text-red-400";
                    statusText.innerText = "Terjadi Kesalahan Server";
                    statusText.classList.add('text-red-400');
                    statusSubtext.innerText = err.message;
                    jsonResult.innerText = err.message;
                    resultSection.classList.remove('hide');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="ph ph-crosshair text-xl mr-2"></i><span>Mulai Pemindaian Objek</span>';
                }
            });

            // 3. Fungsi Mewarnai Canvas (Bounding Box)
            function drawBoundingBoxes(predictions) {
                // Menentukan ketebalan garis dan font berdasarkan resolusi gambar agar proporsional
                const scaleFactor = Math.max(1, outputCanvas.width / 800);
                const lineWidth = Math.max(2, 4 * scaleFactor);
                const fontSize = Math.max(12, 18 * scaleFactor);
                const paddingY = Math.max(24, 30 * scaleFactor);

                predictions.forEach(obj => {
                    const [x, y, width, height] = obj.bbox;
                    const accuracyText = \`\${Math.round(obj.score * 100)}%\`;
                    const labelText = \`\${obj.class.toUpperCase()} [\${accuracyText}]\`;

                    // Warna Kotak: Neon Cyan
                    const strokeColor = '#22d3ee'; 
                    const fillColor = 'rgba(34, 211, 238, 0.2)'; // Transparan cyan di dalam
                    
                    // Isi transparan di dalam kotak
                    ctx.fillStyle = fillColor;
                    ctx.fillRect(x, y, width, height);

                    // Garis luar kotak
                    ctx.strokeStyle = strokeColor;
                    ctx.lineWidth = lineWidth;
                    ctx.strokeRect(x, y, width, height);
                    
                    // Background untuk teks
                    ctx.font = \`bold \${fontSize}px "JetBrains Mono", monospace\`;
                    const textWidth = ctx.measureText(labelText).width;
                    
                    ctx.fillStyle = strokeColor;
                    // Gambar background kotak teks di pojok kiri atas bounding box
                    ctx.fillRect(x - (lineWidth/2), y - paddingY, textWidth + (16 * scaleFactor), paddingY);
                    
                    // Teks label
                    ctx.fillStyle = '#0f172a'; // Warna gelap untuk teks di atas background cyan
                    ctx.fillText(labelText, x + (8 * scaleFactor), y - (paddingY/3.5));
                });
            }

            // 4. Fungsi membuat Badge/Pills List Objek
            function renderBadges(predictions) {
                let html = '';
                predictions.forEach(obj => {
                    const acc = (obj.score * 100).toFixed(0);
                    html += \`
                        <div class="inline-flex items-center px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 hover:border-brand-500 hover:bg-slate-700/80 transition-colors shadow-sm">
                            <span class="w-2 h-2 rounded-full bg-brand-400 mr-2 animate-pulse"></span>
                            <span class="text-sm font-semibold text-slate-200 capitalize mr-2">\${obj.class}</span>
                            <span class="text-xs font-mono font-bold text-brand-400 bg-brand-900/50 px-1.5 py-0.5 rounded">\${acc}%</span>
                        </div>
                    \`;
                });
                detectedBadges.innerHTML = html;
            }
        </script>
    </body>
    </html>
  `);
});

app.listen(PORT, () => console.log(`🚀 AI Object Finder berjalan di http://localhost:${PORT}`));
