// script.js

const inputSection = document.getElementById('input-section');
const uploadForm = document.getElementById('upload-form');
const webcamForm = document.getElementById('webcam-form');
const processSection = document.getElementById('process-section');
const resultSection = document.getElementById('result-section');
const webcamFeed = document.getElementById('webcamFeed');
const webcamCanvas = document.getElementById('webcamCanvas');
const detectionProgress = document.getElementById('detection-progress');

let currentStream;

// --- FUNGSI TAMPILAN UTAMA ---

function showUploadForm() {
    // Sembunyikan semua kecuali form upload
    webcamForm.style.display = 'none';
    uploadForm.style.display = 'block';
    if (currentStream) stopWebcam();
}

function showWebcamForm() {
    // Sembunyikan semua kecuali form webcam
    uploadForm.style.display = 'none';
    webcamForm.style.display = 'block';
    startWebcam();
}

function refreshPage() {
    window.location.reload();
}

// --- LOGIKA WEBCAM ---

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        currentStream = stream;
        webcamFeed.srcObject = stream;
        document.getElementById('bounding-box').style.display = 'block';

    } catch (err) {
        console.error("Gagal mengakses webcam: ", err);
        // Jangan gunakan alert
        webcamForm.style.display = 'none';
        uploadForm.style.display = 'block'; // Kembali ke form upload
        // Di masa depan, kita bisa menambahkan pesan error di HTML
    }
}

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        webcamFeed.srcObject = null;
        currentStream = null;
    }
    document.getElementById('bounding-box').style.display = 'none';
    webcamForm.style.display = 'none';
    // Catatan: Biarkan form upload (jika aktif) tetap terlihat, atau sembunyikan jika dipanggil dari refreshPage
}

function captureAndSubmit() {
    const video = webcamFeed;
    const canvas = webcamCanvas;

    // Tentukan ukuran untuk crop (sesuai area bounding box 75%)
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    const cropFraction = 0.75; 
    const cropX = (videoWidth - videoWidth * cropFraction) / 2;
    const cropY = (videoHeight - videoHeight * cropFraction) / 2;
    const cropWidth = videoWidth * cropFraction;
    const cropHeight = videoHeight * cropFraction;

    canvas.width = cropWidth;
    canvas.height = cropHeight;
    const ctx = canvas.getContext('2d');

    // Gambar area crop dari video ke canvas
    ctx.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

    // Konversi canvas ke base64 Data URL
    const capturedImage = canvas.toDataURL('image/jpeg');

    // Matikan webcam
    stopWebcam();
    
    // --- FIX ---
    // Asumsi: Webcam hanya untuk kulit, bukan 'brain-mri'
    sendToBackendForDetection(capturedImage, 'skin-photo'); 
}

// --- LOGIKA SUBMIT & PROSES (INTEGRASI BACKEND SEBENARNYA) ---

function processDetection() {
    const fileInput = document.getElementById('imageUpload');
    const typeSelect = document.getElementById('diagnosisType');
    
    if (fileInput.files.length === 0) {
        // Jangan gunakan alert
        console.warn("Mohon unggah file gambar terlebih dahulu.");
        return;
    }
    if (!typeSelect.value) {
        console.warn("Mohon pilih jenis citra diagnosis.");
        return;
    }
    
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        // Konversi file ke base64
        const imageData = e.target.result;
        sendToBackendForDetection(imageData, typeSelect.value);
    };

    reader.readAsDataURL(file);
}

// ** FUNGSI PENTING: MENGIRIM DATA KE BACKEND **
function sendToBackendForDetection(imageData, diagnosisType) {
    // 1. Tampilkan Progress Section
    inputSection.style.display = 'none';
    processSection.style.display = 'block';
    
    // Atur progress bar sebagai loading tak terbatas (animated)
    detectionProgress.style.width = '100%';
    detectionProgress.classList.add('progress-bar-animated');
    detectionProgress.innerHTML = 'Mengirim & Memproses...';

    // 2. Data yang akan dikirim ke API backend
    const dataToSend = {
        image: imageData, // Data gambar dalam format base64
        type: diagnosisType
    };

    const backendUrl = '/detect'; 

    // 3. Panggil API backend menggunakan fetch
    // --- FIX ---
    // Logika fetch Anda sebelumnya salah.
    // Ini adalah cara yang benar untuk mengirim JSON.
    fetch(backendUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
    .then(response => {
        // Hentikan animasi progress bar
        detectionProgress.classList.remove('progress-bar-animated');
        
        if (!response.ok) {
            // Jika status HTTP bukan 200-299
            return response.json().then(errData => {
                 throw new Error(errData.message || `Deteksi Gagal! Status HTTP: ${response.status}`);
            }).catch(() => {
                throw new Error(`Deteksi Gagal! Status HTTP: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(result => {
        // 4. Tampilkan hasil dari backend
        if (result.success) {
            displayResult(result);
        } else {
            throw new Error(result.message || "Backend mengembalikan error tanpa pesan.");
        }
    })
    .catch(error => {
        console.error('Error saat mengirim ke backend:', error);
        
        // Menampilkan pesan error dan kembali ke input
        processSection.style.display = 'none';
        inputSection.style.display = 'block';
        // Di masa depan, gunakan elemen error HTML, bukan alert
        console.error(`Terjadi Kesalahan: ${error.message}.`);
        
        // Reset progress bar
        detectionProgress.style.width = '0%';
        detectionProgress.innerHTML = '0%';
    });
}

function displayResult(result) {
    // 1. Sembunyikan Progress Bar, Tampilkan Hasil
    processSection.style.display = 'none';
    resultSection.style.display = 'block';

    // 2. Isi data hasil ke elemen HTML (sesuai format JSON dari backend)
    document.getElementById('output-image').src = result.image;
    document.getElementById('res-type').textContent = (result.type === 'brain-mri' ? 'MRI Otak' : 'Foto Kulit');
    document.getElementById('res-classification').textContent = result.classification;
    // Backend mengirim float (mis 0.95), JS mengubahnya ke persen
    document.getElementById('res-confidence').textContent = (parseFloat(result.confidence) * 100).toFixed(1) + '%';
    document.getElementById('res-size').textContent = result.size;
    document.getElementById('res-explanation').textContent = "Penjelasan Singkat: " + result.explanation;
    
    // --- NEW: Handle Mask Image Display ---
    const maskImage = document.getElementById('mask-image');
    const maskTitle = document.getElementById('mask-title');
    if (result.mask_image_url) {
        maskImage.src = result.mask_image_url;
        maskImage.style.display = 'block';
    } else {
        maskImage.style.display = 'none'; // Sembunyikan jika bukan brain scan
        maskTitle.style.display = 'none';
    }

    // Tambahkan warna pada klasifikasi berdasarkan hasil
    const classificationElement = document.getElementById('res-classification');
    const classificationText = classificationElement.textContent.toLowerCase();
    
    if (classificationText.includes('normal') || classificationText.includes('no tumor')) {
        classificationElement.className = 'text-success fw-bold';
    } else {
        classificationElement.className = 'text-danger fw-bold';
    }
}