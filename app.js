import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── DOM ───────────────────────────────────────────────────────────────────────
const hero          = document.getElementById("hero");
const app           = document.getElementById("app");
const video         = document.getElementById("video");
const canvas        = document.getElementById("canvas");
const ctx           = canvas.getContext("2d", { willReadFrequently: true });
const loadingOverlay= document.getElementById("loadingOverlay");
const btnStart      = document.getElementById("btnStart");
const btnBack       = document.getElementById("btnBack");
const btnSwitch     = document.getElementById("btnSwitch");
const loadingText   = document.getElementById("loadingText");
const progressBar   = document.getElementById("progressBar");
const helpOverlay   = document.getElementById("helpOverlay");
const helpClose     = document.getElementById("helpClose");

// ── Resolusi proses ───────────────────────────────────────────────────────────
const PROC_W = 640;
const PROC_H = 360;

const offCanvas  = new OffscreenCanvas(PROC_W, PROC_H);
const offCtx     = offCanvas.getContext("2d", { willReadFrequently: true });
const maskCanvas = new OffscreenCanvas(PROC_W, PROC_H);
const maskCtx    = maskCanvas.getContext("2d");
let fxBuf        = null;

// ── Daftar efek cycle ─────────────────────────────────────────────────────────
const EFFECT_LIST = ["invert", "thermal", "noir", "glitch", "pixelate", "emboss", "edge", "vhs", "mirror", "kaleidoscope", "rgbsplit", "sketch"];

// Nama efek dalam bahasa Indonesia
const EFFECT_NAMES = {
  invert: "Invers",
  thermal: "Termal",
  noir: "Hitam Putih",
  glitch: "Glitch",
  pixelate: "Piksel",
  emboss: "Timbul",
  edge: "Garis Tepi",
  vhs: "VHS Retro",
  mirror: "Cermin",
  kaleidoscope: "Kaleidoskop",
  rgbsplit: "RGB Split",
  sketch: "Sketsa",
};

// ── State persegi ─────────────────────────────────────────────────────────────
const box1 = { active: false, effectIdx: 0, triggered: false, lastToggle: 0 };
const box2 = { active: false, effectIdx: 0, triggered: false, lastToggle: 0 };

let cycleTriggered = false;
let cycleLastTime  = 0;
let cycleHoldStart = 0;  // Waktu mulai tahan gesture
let cycleAutoMode  = false;  // Mode auto-cycle saat ditahan
const CYCLE_AUTO_INTERVAL = 300;  // Auto cycle tiap 300ms saat ditahan

// ── State prisma segi lima ────────────────────────────────────────────────────
const PRISM_COOLDOWN = 1000;
const prism = {
  active:      false,
  triggered:   false,
  lastToggle:  0,
  sideEffects: [],   // efek random untuk tiap 5 sisi, di-assign saat aktif
};

// ── State face blur ───────────────────────────────────────────────────────────
let faceBlurActive   = false;       // toggle on/off
let metalTriggered   = false;       // untuk deteksi rising edge
let metalLastTime    = 0;
let lastFaceLandmarks = [];         // cache landmark wajah
let faceFrameCount   = 0;
const FACE_UPDATE_EVERY = 5;        // Update setiap 5 frame (lebih jarang)

const PINCH_THRESHOLD = 0.15;
const COOLDOWN_MS     = 300;  // Kurangi dari 800ms ke 300ms untuk bisa spam
const METAL_COOLDOWN  = 1000;
const MENU_COOLDOWN   = 600;

let handLandmarker = null;
let faceLandmarker = null;
let animId         = null;
let currentFacing  = "user";
let lastTimestamp  = 0;

// ── Landmark smoothing buffer (2 frame moving average) ────────────────────────
// Set ke 1 untuk disable smoothing (response tercepat tapi ada jitter)
// Set ke 2-3 untuk balance antara smooth dan responsive (recommended)
const SMOOTH_FRAMES = 1;  // 1 = no smoothing (fastest response)
const landmarkHistory = { Left: [], Right: [] };

function smoothLandmarks(rawHands) {
  const smoothed = {};
  
  for (const side of ['Left', 'Right']) {
    if (!rawHands[side]) {
      // Reset history jika tangan hilang
      landmarkHistory[side] = [];
      continue;
    }
    
    // Tambah landmark terbaru ke history
    landmarkHistory[side].push(rawHands[side]);
    if (landmarkHistory[side].length > SMOOTH_FRAMES) {
      landmarkHistory[side].shift();
    }
    
    // Hitung rata-rata dari semua frame dalam buffer
    const history = landmarkHistory[side];
    const avgLandmarks = [];
    
    for (let i = 0; i < rawHands[side].length; i++) {
      let sumX = 0, sumY = 0, sumZ = 0;
      
      for (const frame of history) {
        sumX += frame[i].x;
        sumY += frame[i].y;
        sumZ += frame[i].z || 0;
      }
      
      avgLandmarks.push({
        x: sumX / history.length,
        y: sumY / history.length,
        z: sumZ / history.length,
      });
    }
    
    smoothed[side] = avgLandmarks;
  }
  
  return smoothed;
}

// ── Loading progress helper ───────────────────────────────────────────────────
function updateLoadingProgress(percent, text) {
  if (progressBar) progressBar.style.width = `${percent}%`;
  if (loadingText) loadingText.textContent = text;
}

// ── Deteksi gestur: jempol + jari manis (ring finger) ────────────────────────
// Bahasa isyarat: gesture ini sering dipakai untuk "love" atau "OK"
function isRingPinch(lm) {
  return pinchDist(lm, 16) < PINCH_THRESHOLD;  // landmark 16 = ring finger tip
}

// ── Deteksi gestur PEACE (✌️) — telunjuk + tengah tegak, sisanya menekuk ─────
function isPeace(lm) {
  const up   = (tip, pip) => lm[tip].y < lm[pip].y;
  const down = (tip, pip) => lm[tip].y > lm[pip].y;
  // Telunjuk (8) dan tengah (12) harus tegak, manis (16) dan kelingking (20) menekuk
  return up(8, 6) && up(12, 10) && down(16, 14) && down(20, 18);
}

// ── Deteksi gestur FIST (✊) — semua jari menekuk ke dalam ────────────────────
function isFist(lm) {
  const down = (tip, pip) => lm[tip].y > lm[pip].y;
  // Semua jari (telunjuk, tengah, manis, kelingking) harus menekuk
  return down(8, 6) && down(12, 10) && down(16, 14) && down(20, 18);
}

// ── Deteksi pointing gesture — hanya telunjuk tegak ──────────────────────────
function isPointing(lm) {
  const up   = (tip, pip) => lm[tip].y < lm[pip].y;
  const down = (tip, pip) => lm[tip].y > lm[pip].y;
  // Hanya telunjuk (8) tegak, sisanya menekuk
  return up(8, 6) && down(12, 10) && down(16, 14) && down(20, 18);
}

// ── Buttons ───────────────────────────────────────────────────────────────────
btnStart.addEventListener("click", async () => {
  hero.classList.add("hidden");
  app.classList.remove("hidden");
  
  try {
    updateLoadingProgress(0, "Memuat kamera...");
    await initCamera();

    updateLoadingProgress(30, "Memuat model...");
    await initModels();

    updateLoadingProgress(100, "Siap!");
    await new Promise(r => setTimeout(r, 300));

    loadingOverlay.classList.add("done");
    loop();
  } catch (error) {
    console.error("Error during initialization:", error);
    updateLoadingProgress(0, "Gagal memuat. Coba lagi.");
    await new Promise(r => setTimeout(r, 2000));
    app.classList.add("hidden");
    hero.classList.remove("hidden");
  }
});
btnBack.addEventListener("click", () => {
  if (animId) cancelAnimationFrame(animId);
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  box1.active = box2.active = false;
  prism.active = false;
  faceBlurActive = false;
  loadingOverlay.classList.remove("done");
  app.classList.add("hidden");
  hero.classList.remove("hidden");
});
btnSwitch.addEventListener("click", async () => {
  if (!video.srcObject) return;
  video.srcObject.getTracks().forEach(t => t.stop());
  currentFacing = currentFacing === "user" ? "environment" : "user";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: currentFacing, width: { ideal: 640 }, height: { ideal: 360 } },
  });
  video.srcObject = stream;
  await video.play();
});

// ── Help overlay ──────────────────────────────────────────────────────────────
helpClose.addEventListener("click", () => {
  helpOverlay.classList.add("hidden");
});

document.getElementById("btnHelp").addEventListener("click", () => {
  helpOverlay.classList.toggle("hidden");
});

// ── Camera ────────────────────────────────────────────────────────────────────
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: currentFacing, width: { ideal: 640 }, height: { ideal: 360 } },
  });
  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });
  await video.play();
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  fxBuf = new ImageData(PROC_W, PROC_H);
}

// ── Models ────────────────────────────────────────────────────────────────────
// Hanya load hand landmarker saat start — face landmarker di-lazy load saat dibutuhkan
async function initModels() {
  updateLoadingProgress(40, "Memuat model...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  // Simpan vision resolver agar bisa dipakai ulang oleh ensureFaceLandmarker
  window._visionResolver = vision;

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode:                "VIDEO",
      numHands:                   2,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence:  0.7,
      minTrackingConfidence:      0.7,
    });
  } catch (e) {
    console.warn("GPU gagal untuk hand, fallback ke CPU:", e);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU",
      },
      runningMode:                "VIDEO",
      numHands:                   2,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence:  0.7,
      minTrackingConfidence:      0.7,
    });
  }

  console.log("Hand model loaded.");
}

// ── Lazy load face model — dipanggil hanya saat gesture metal pertama kali ────
let faceLoading = false; // cegah double-load
async function ensureFaceLandmarker() {
  if (faceLandmarker) return true;  // sudah ada
  if (faceLoading)    return false; // sedang loading, skip frame ini
  faceLoading = true;

  const vision = window._visionResolver;
  if (!vision) { faceLoading = false; return false; }

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode:               "VIDEO",
      numFaces:                  2,
      minFaceDetectionConfidence: 0.6,
      minFacePresenceConfidence:  0.6,
      minTrackingConfidence:      0.6,
    });
  } catch (e) {
    console.warn("GPU gagal untuk face, fallback ke CPU:", e);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "CPU",
      },
      runningMode:               "VIDEO",
      numFaces:                  2,
      minFaceDetectionConfidence: 0.6,
      minFacePresenceConfidence:  0.6,
      minTrackingConfidence:      0.6,
    });
  }

  faceLoading = false;
  console.log("Face model loaded (lazy).");
  return true;
}

// ── Pinch distance ────────────────────────────────────────────────────────────
function pinchDist(lm, fingerId) {
  const dx = lm[4].x - lm[fingerId].x;
  const dy = lm[4].y - lm[fingerId].y;
  const px = lm[0].x - lm[5].x;
  const py = lm[0].y - lm[5].y;
  const palm = Math.sqrt(px*px + py*py);
  return palm === 0 ? Infinity : Math.sqrt(dx*dx + dy*dy) / palm;
}

// ── Jarak absolut antara dua ujung jari (normalized koordinat) ────────────────
// Dipakai untuk deteksi "semua jari kiri bersentuhan dengan jari kanan"
function fingertipDist(lmA, lmB, idx) {
  const dx = lmA[idx].x - lmB[idx].x;
  const dy = lmA[idx].y - lmB[idx].y;
  // Normalisasi dengan ukuran telapak tangan rata-rata keduanya
  const palmA = Math.sqrt((lmA[0].x-lmA[5].x)**2 + (lmA[0].y-lmA[5].y)**2);
  const palmB = Math.sqrt((lmB[0].x-lmB[5].x)**2 + (lmB[0].y-lmB[5].y)**2);
  const palm  = (palmA + palmB) / 2;
  return palm === 0 ? Infinity : Math.sqrt(dx*dx + dy*dy) / palm;
}

// ── Deteksi gesture: semua jari kiri menyentuh jari pasangannya di tangan kanan
function isAllFingertipsTouch(L, R) {
  const TOUCH = 0.35; // threshold lebih longgar karena pakai koordinat absolut
  return fingertipDist(L, R,  4) < TOUCH  // jempol-jempol
      && fingertipDist(L, R,  8) < TOUCH  // telunjuk-telunjuk
      && fingertipDist(L, R, 12) < TOUCH  // tengah-tengah
      && fingertipDist(L, R, 16) < TOUCH  // manis-manis
      && fingertipDist(L, R, 20) < TOUCH; // kelingking-kelingking
}

// ── Deteksi gestur METAL ──────────────────────────────────────────────────────
function isMetal(lm) {
  const up   = (tip, pip) => lm[tip].y < lm[pip].y;
  const down = (tip, pip) => lm[tip].y > lm[pip].y;
  return up(8, 6) && up(20, 18) && down(12, 10) && down(16, 14);
}

// ── Blur wajah — ellipse adaptif dari bounding box semua landmark ─────────────
// Lebih robust dari kontur tetap: bekerja untuk wajah depan, samping, miring.
function blurFacesOval(faceLandmarksList, DW, DH) {
  if (!faceLandmarksList.length) return;

  for (const landmarks of faceLandmarksList) {
    // Hitung bounding box dari semua landmark yang ada
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const lm of landmarks) {
      const x = (1 - lm.x) * DW;
      const y = lm.y * DH;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    // Pusat dan radius ellipse dengan sedikit padding
    const cx  = (minX + maxX) / 2;
    const cy  = (minY + maxY) / 2;
    const rx  = (maxX - minX) / 2 * 1.15;  // padding 15% horizontal
    const ry  = (maxY - minY) / 2 * 1.10;  // padding 10% vertikal

    // Clip ke ellipse, gambar ulang video dengan blur
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.clip();
    ctx.filter = "blur(20px)";
    ctx.translate(DW, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, DW, DH);
    ctx.restore();
  }
}

// ── Landmark → pixel coords (mirror X) ───────────────────────────────────────
function lmPx(lm, idx) {
  return { x: (1 - lm[idx].x) * PROC_W, y: lm[idx].y * PROC_H };
}

// ── Visual feedback: glow di jari saat gesture aktif ─────────────────────────
function drawFingerGlow(hand, fingerIdx, color, DW, DH, label = null) {
  if (!hand) return;
  
  const x = (1 - hand[fingerIdx].x) * DW;
  const y = hand[fingerIdx].y * DH;
  
  // Outer glow
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 20;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fill();
  
  // Inner circle
  ctx.shadowBlur = 8;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  
  // Label text (optional)
  if (label) {
    ctx.save();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 4;
    ctx.fillText(label, x, y - 20);
    ctx.restore();
  }
}

function drawPinchGlow(hand, thumbIdx, fingerIdx, color, DW, DH) {
  if (!hand) return;
  
  const x1 = (1 - hand[thumbIdx].x) * DW;
  const y1 = hand[thumbIdx].y * DH;
  const x2 = (1 - hand[fingerIdx].x) * DW;
  const y2 = hand[fingerIdx].y * DH;
  
  // Line between fingers
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowColor = color;
  ctx.shadowBlur = 15;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
  
  // Glow at both fingertips
  drawFingerGlow(hand, thumbIdx, color, DW, DH);
  drawFingerGlow(hand, fingerIdx, color, DW, DH);
}

// ── Effects ───────────────────────────────────────────────────────────────────
function applyInvert(src, dst) {
  const s = src.data, d = dst.data;
  for (let i = 0; i < s.length; i += 4) {
    d[i] = 255-s[i]; d[i+1] = 255-s[i+1]; d[i+2] = 255-s[i+2]; d[i+3] = 255;
  }
}

const INFERNO = buildColormap([
  [0,0,4],[20,11,53],[58,9,99],[96,19,110],[133,33,107],
  [169,46,94],[203,65,73],[229,89,52],[248,125,21],[254,167,10],
  [252,209,22],[252,255,164],
]);
function buildColormap(stops) {
  const lut = new Uint8Array(256*3), n = stops.length-1;
  for (let i = 0; i < 256; i++) {
    const t = i/255*n, idx = Math.min(n-1, Math.floor(t)), f = t-idx;
    const a = stops[idx], b = stops[idx+1];
    lut[i*3]   = a[0]+(b[0]-a[0])*f;
    lut[i*3+1] = a[1]+(b[1]-a[1])*f;
    lut[i*3+2] = a[2]+(b[2]-a[2])*f;
  }
  return lut;
}

function applyThermal(src, dst) {
  const s = src.data, d = dst.data;
  for (let i = 0; i < s.length; i += 4) {
    const g = (s[i]*0.299 + s[i+1]*0.587 + s[i+2]*0.114)|0;
    d[i] = INFERNO[g*3]; d[i+1] = INFERNO[g*3+1]; d[i+2] = INFERNO[g*3+2]; d[i+3] = 255;
  }
}

let vigCache = null;
function getVignette() {
  if (vigCache) return vigCache;
  vigCache = new Float32Array(PROC_W * PROC_H);
  const cx = PROC_W/2, cy = PROC_H/2;
  for (let y = 0; y < PROC_H; y++)
    for (let x = 0; x < PROC_W; x++)
      vigCache[y*PROC_W+x] = Math.max(0.2, Math.min(1.0,
        1.0 - Math.sqrt(((x-cx)/cx)**2 + ((y-cy)/cy)**2) * 0.8));
  return vigCache;
}
function applyNoir(src, dst) {
  const s = src.data, d = dst.data, vign = getVignette();
  for (let i = 0; i < s.length; i += 4) {
    const c = Math.min(255, Math.max(0,
      (s[i]*0.299 + s[i+1]*0.587 + s[i+2]*0.114 - 60) * 1.6)) * vign[i>>2];
    d[i] = d[i+1] = d[i+2] = c; d[i+3] = 255;
  }
}

function applyGlitch(src, dst, ts) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H, shift = 6;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i  = (y*w+x)*4;
      const iR = (y*w+Math.min(w-1, x+shift))*4;
      const iB = (y*w+Math.max(0,   x-shift))*4;
      d[i] = s[iR]; d[i+1] = s[i+1]; d[i+2] = s[iB+2]; d[i+3] = 255;
    }
  }
  const seed = Math.floor(ts/80);
  for (let n = 0; n < 4; n++) {
    const y   = Math.abs(Math.sin(seed*7.3+n*13.7)*h)|0;
    const off = (Math.sin(seed*3.1+n*5.9)*15)|0;
    const row = new Uint8Array(w*4);
    for (let x = 0; x < w; x++) {
      const s2 = (y*w+Math.min(w-1, Math.max(0, x+off)))*4;
      row[x*4] = d[s2]; row[x*4+1] = d[s2+1]; row[x*4+2] = d[s2+2]; row[x*4+3] = 255;
    }
    dst.data.set(row, y*w*4);
  }
}

// ── Pixelate effect (mosaic) ──────────────────────────────────────────────────
function applyPixelate(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  const blockSize = 12;
  
  for (let by = 0; by < h; by += blockSize) {
    for (let bx = 0; bx < w; bx += blockSize) {
      let r = 0, g = 0, b = 0, count = 0;
      
      // Hitung rata-rata warna dalam blok
      for (let dy = 0; dy < blockSize && by + dy < h; dy++) {
        for (let dx = 0; dx < blockSize && bx + dx < w; dx++) {
          const i = ((by + dy) * w + (bx + dx)) * 4;
          r += s[i]; g += s[i+1]; b += s[i+2];
          count++;
        }
      }
      
      r = (r/count)|0; g = (g/count)|0; b = (b/count)|0;
      
      // Isi blok dengan warna rata-rata
      for (let dy = 0; dy < blockSize && by + dy < h; dy++) {
        for (let dx = 0; dx < blockSize && bx + dx < w; dx++) {
          const i = ((by + dy) * w + (bx + dx)) * 4;
          d[i] = r; d[i+1] = g; d[i+2] = b; d[i+3] = 255;
        }
      }
    }
  }
}

// ── Emboss effect (3D relief) ─────────────────────────────────────────────────
function applyEmboss(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  
  for (let y = 1; y < h-1; y++) {
    for (let x = 1; x < w-1; x++) {
      const i = (y*w+x)*4;
      const iUL = ((y-1)*w+(x-1))*4;  // upper-left
      const iBR = ((y+1)*w+(x+1))*4;  // bottom-right
      
      const diffR = s[iUL] - s[iBR] + 128;
      const diffG = s[iUL+1] - s[iBR+1] + 128;
      const diffB = s[iUL+2] - s[iBR+2] + 128;
      
      d[i]   = Math.max(0, Math.min(255, diffR));
      d[i+1] = Math.max(0, Math.min(255, diffG));
      d[i+2] = Math.max(0, Math.min(255, diffB));
      d[i+3] = 255;
    }
  }
}

// ── Edge detection (Sobel operator) ───────────────────────────────────────────
function applyEdge(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  
  for (let y = 1; y < h-1; y++) {
    for (let x = 1; x < w-1; x++) {
      const i = (y*w+x)*4;
      
      // Sobel kernels
      const gx = 
        -s[((y-1)*w+(x-1))*4] + s[((y-1)*w+(x+1))*4] +
        -2*s[(y*w+(x-1))*4] + 2*s[(y*w+(x+1))*4] +
        -s[((y+1)*w+(x-1))*4] + s[((y+1)*w+(x+1))*4];
      
      const gy = 
        -s[((y-1)*w+(x-1))*4] - 2*s[((y-1)*w+x)*4] - s[((y-1)*w+(x+1))*4] +
        s[((y+1)*w+(x-1))*4] + 2*s[((y+1)*w+x)*4] + s[((y+1)*w+(x+1))*4];
      
      const mag = Math.sqrt(gx*gx + gy*gy);
      const val = Math.min(255, mag);
      
      d[i] = d[i+1] = d[i+2] = 255 - val;  // invert untuk edge putih di background hitam
      d[i+3] = 255;
    }
  }
}

// ── VHS retro effect (scanlines + color shift) ────────────────────────────────
function applyVHS(src, dst, ts) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  const seed = Math.floor(ts/100);
  
  for (let y = 0; y < h; y++) {
    // Scanline effect
    const scanline = (y % 3 === 0) ? 0.85 : 1.0;
    
    // Random horizontal shift per scanline (glitch effect)
    const shift = (Math.abs(Math.sin(seed*0.1 + y*0.3)) > 0.98) 
      ? ((Math.sin(y*13.7+seed)*5)|0) 
      : 0;
    
    for (let x = 0; x < w; x++) {
      const i = (y*w+x)*4;
      const sx = Math.max(0, Math.min(w-1, x + shift));
      const si = (y*w+sx)*4;
      
      // Color shift + desaturate + scanlines
      d[i]   = (s[si]*0.9 + 30) * scanline;
      d[i+1] = (s[si+1]*0.8 + 20) * scanline;
      d[i+2] = (s[si+2]*0.7 + 40) * scanline;
      d[i+3] = 255;
    }
  }
  
  // Tambah noise
  for (let i = 0; i < d.length; i += 4) {
    if (Math.random() > 0.97) {
      const noise = (Math.random()*80)|0;
      d[i] = d[i+1] = d[i+2] = noise;
    }
  }
}

// ── Mirror kaleidoscope effect ────────────────────────────────────────────────
function applyMirror(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  const hw = w >> 1, hh = h >> 1;
  
  // Quadrant 1: copy original top-left
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < hw; x++) {
      const i = (y*w+x)*4;
      d[i] = s[i]; d[i+1] = s[i+1]; d[i+2] = s[i+2]; d[i+3] = 255;
    }
  }
  
  // Quadrant 2: mirror horizontal from Q1
  for (let y = 0; y < hh; y++) {
    for (let x = hw; x < w; x++) {
      const i = (y*w+x)*4;
      const si = (y*w+(w-1-x))*4;
      d[i] = s[si]; d[i+1] = s[si+1]; d[i+2] = s[si+2]; d[i+3] = 255;
    }
  }
  
  // Quadrant 3 & 4: mirror vertical dari Q1 & Q2
  for (let y = hh; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y*w+x)*4;
      const si = ((h-1-y)*w+x)*4;
      d[i] = d[si]; d[i+1] = d[si+1]; d[i+2] = d[si+2]; d[i+3] = 255;
    }
  }
}

// ── Kaleidoscope radial mirror effect ─────────────────────────────────────────
function applyKaleidoscope(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  const cx = w / 2, cy = h / 2;
  const segments = 6; // 6 segmen radial
  const angleStep = (Math.PI * 2) / segments;
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += Math.PI * 2;
      
      // Mirror dalam segment
      const segmentIdx = Math.floor(angle / angleStep);
      const segmentAngle = angle - segmentIdx * angleStep;
      const mirroredAngle = segmentIdx % 2 === 0 
        ? segmentAngle 
        : angleStep - segmentAngle;
      
      const finalAngle = segmentIdx * angleStep + mirroredAngle;
      
      // Source coordinate
      const sx = Math.round(cx + dist * Math.cos(finalAngle));
      const sy = Math.round(cy + dist * Math.sin(finalAngle));
      
      const i = (y*w+x)*4;
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        const si = (sy*w+sx)*4;
        d[i] = s[si]; d[i+1] = s[si+1]; d[i+2] = s[si+2]; d[i+3] = 255;
      } else {
        d[i] = d[i+1] = d[i+2] = 0; d[i+3] = 255;
      }
    }
  }
}

// ── RGB split chromatic aberration effect ─────────────────────────────────────
function applyRGBSplit(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  const offsetR = 5;  // Red channel shift right
  const offsetB = -5; // Blue channel shift left
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y*w+x)*4;
      
      // Red channel (shifted right)
      const xR = Math.min(w-1, Math.max(0, x + offsetR));
      const iR = (y*w+xR)*4;
      d[i] = s[iR];
      
      // Green channel (no shift)
      d[i+1] = s[i+1];
      
      // Blue channel (shifted left)
      const xB = Math.min(w-1, Math.max(0, x + offsetB));
      const iB = (y*w+xB)*4;
      d[i+2] = s[iB+2];
      
      d[i+3] = 255;
    }
  }
}

// ── Sketch/cartoon effect (edge + posterize) ──────────────────────────────────
function applySketch(src, dst) {
  const s = src.data, d = dst.data, w = PROC_W, h = PROC_H;
  
  // Step 1: Edge detection (simplified Sobel)
  const edges = new Uint8Array(w * h);
  for (let y = 1; y < h-1; y++) {
    for (let x = 1; x < w-1; x++) {
      const i = (y*w+x)*4;
      
      const gx = 
        -s[((y-1)*w+(x-1))*4] + s[((y-1)*w+(x+1))*4] +
        -2*s[(y*w+(x-1))*4] + 2*s[(y*w+(x+1))*4] +
        -s[((y+1)*w+(x-1))*4] + s[((y+1)*w+(x+1))*4];
      
      const gy = 
        -s[((y-1)*w+(x-1))*4] - 2*s[((y-1)*w+x)*4] - s[((y-1)*w+(x+1))*4] +
        s[((y+1)*w+(x-1))*4] + 2*s[((y+1)*w+x)*4] + s[((y+1)*w+(x+1))*4];
      
      const mag = Math.sqrt(gx*gx + gy*gy);
      edges[y*w+x] = mag > 80 ? 255 : 0;
    }
  }
  
  // Step 2: Posterize colors (reduce to 4 levels per channel)
  const levels = 4;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y*w+x)*4;
      const edge = edges[y*w+x];
      
      if (edge > 128) {
        // Strong edge = black line
        d[i] = d[i+1] = d[i+2] = 0;
      } else {
        // Posterize
        d[i]   = Math.floor(s[i]   / 256 * levels) * (255 / (levels-1));
        d[i+1] = Math.floor(s[i+1] / 256 * levels) * (255 / (levels-1));
        d[i+2] = Math.floor(s[i+2] / 256 * levels) * (255 / (levels-1));
      }
      d[i+3] = 255;
    }
  }
}

const EFFECT_FNS    = {
  invert: applyInvert,
  thermal: applyThermal,
  noir: applyNoir,
  glitch: (src, dst) => applyGlitch(src, dst, lastTimestamp),
  pixelate: applyPixelate,
  emboss: applyEmboss,
  edge: applyEdge,
  vhs: (src, dst) => applyVHS(src, dst, lastTimestamp),
  mirror: applyMirror,
  kaleidoscope: applyKaleidoscope,
  rgbsplit: applyRGBSplit,
  sketch: applySketch,
};
const BORDER_COLORS = {
  invert: "rgba(255,255,255,0.8)",
  thermal: "rgba(255,120,60,0.9)",
  noir: "rgba(200,200,200,0.8)",
  glitch: "rgba(0,255,80,0.9)",
  pixelate: "rgba(255,0,220,0.85)",
  emboss: "rgba(180,140,255,0.85)",
  edge: "rgba(0,200,255,0.9)",
  vhs: "rgba(255,100,150,0.85)",
  mirror: "rgba(100,255,200,0.9)",
  kaleidoscope: "rgba(255,50,255,0.9)",
  rgbsplit: "rgba(0,255,255,0.85)",
  sketch: "rgba(255,200,100,0.85)",
};

// ── Mask + composite ──────────────────────────────────────────────────────────
function buildMask(pts) {
  maskCtx.clearRect(0, 0, PROC_W, PROC_H);
  maskCtx.fillStyle = "#fff";
  maskCtx.beginPath();
  maskCtx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) maskCtx.lineTo(pts[i].x, pts[i].y);
  maskCtx.closePath();
  maskCtx.fill();
  return maskCtx.getImageData(0, 0, PROC_W, PROC_H);
}

function composite(base, fx, mask) {
  const b = base.data, e = fx.data, m = mask.data;
  for (let i = 0; i < b.length; i += 4) {
    const a = m[i]/255;
    b[i]   = b[i]*(1-a)+e[i]*a;
    b[i+1] = b[i+1]*(1-a)+e[i+1]*a;
    b[i+2] = b[i+2]*(1-a)+e[i+2]*a;
  }
}

// ── Border ────────────────────────────────────────────────────────────────────
function drawBorder(pts, color, sx, sy) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.lineJoin    = "round";
  ctx.lineCap     = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur  = 10;
  ctx.beginPath();
  ctx.moveTo(pts[0].x*sx, pts[0].y*sy);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x*sx, pts[i].y*sy);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

// ── Apply region ──────────────────────────────────────────────────────────────
function applyRegion(effectFn, pts, borderColor, sx, sy, ts) {
  const frame = offCtx.getImageData(0, 0, PROC_W, PROC_H);
  fxBuf.data.set(frame.data);
  effectFn(frame, fxBuf);
  const mask = buildMask(pts);
  composite(frame, fxBuf, mask);
  offCtx.putImageData(frame, 0, 0);
  applyRegion._borders.push({ pts, borderColor, sx, sy });
}
applyRegion._borders = [];

function flushToDisplay(DW, DH) {
  // 1. Copy offscreen (berisi semua efek persegi) ke display
  ctx.drawImage(offCanvas, 0, 0, DW, DH);
  // 2. Gambar semua border di atas
  for (const { pts, borderColor, sx, sy } of applyRegion._borders) {
    drawBorder(pts, borderColor, sx, sy);
  }
  applyRegion._borders = [];
}

// ── Pentagon prism 3D renderer ────────────────────────────────────────────────
// 5 persegi, tiap persegi = 4 sudut dari 2 pasang jari bersebelahan
// Persegi 1: jempol kiri(4), jempol kanan(4), telunjuk kanan(8), telunjuk kiri(8)
// Persegi 2: telunjuk kiri(8), telunjuk kanan(8), tengah kanan(12), tengah kiri(12)
// Persegi 3: tengah kiri(12), tengah kanan(12), manis kanan(16), manis kiri(16)
// Persegi 4: manis kiri(16), manis kanan(16), kelingking kanan(20), kelingking kiri(20)
// Persegi 5: kelingking kiri(20), kelingking kanan(20), jempol kanan(4), jempol kiri(4)

function fingerPx(lm, idx, W, H) {
  return { x: (1 - lm[idx].x) * W, y: lm[idx].y * H };
}

function renderPrism(L, R, DW, DH) {
  // Definisi 5 persegi — tiap entry: [idxA, idxB] pasang jari
  // Sudut persegi: [A_kiri, A_kanan, B_kanan, B_kiri]
  const QUAD_PAIRS = [
    [4,  8],   // jempol → telunjuk
    [8,  12],  // telunjuk → tengah
    [12, 16],  // tengah → manis
    [16, 20],  // manis → kelingking
    [20, 4],   // kelingking → jempol (menutup lingkaran)
  ];

  // Siapkan offCanvas dengan frame video terkini + blur wajah jika aktif
  offCtx.save();
  offCtx.translate(PROC_W, 0);
  offCtx.scale(-1, 1);
  offCtx.drawImage(video, 0, 0, PROC_W, PROC_H);
  offCtx.restore();

  // Render blur wajah ke offCanvas sebelum efek persegi
  if (faceBlurActive && lastFaceLandmarks.length > 0) {
    for (const landmarks of lastFaceLandmarks) {
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      for (const lm of landmarks) {
        const x = (1 - lm.x) * PROC_W;
        const y = lm.y * PROC_H;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      const rx = (maxX - minX) / 2 * 1.15, ry = (maxY - minY) / 2 * 1.10;
      offCtx.save();
      offCtx.beginPath();
      offCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      offCtx.clip();
      offCtx.filter = "blur(20px)";
      offCtx.translate(PROC_W, 0);
      offCtx.scale(-1, 1);
      offCtx.drawImage(video, 0, 0, PROC_W, PROC_H);
      offCtx.restore();
    }
  }

  const sx = DW / PROC_W, sy = DH / PROC_H;

  for (let i = 0; i < 5; i++) {
    const [idxA, idxB] = QUAD_PAIRS[i];
    const effectKey = prism.sideEffects[i];
    const effectFn  = EFFECT_FNS[effectKey];
    const color     = BORDER_COLORS[effectKey];

    // 4 sudut dalam proc space
    const AL = fingerPx(L, idxA, PROC_W, PROC_H);
    const AR = fingerPx(R, idxA, PROC_W, PROC_H);
    const BR = fingerPx(R, idxB, PROC_W, PROC_H);
    const BL = fingerPx(L, idxB, PROC_W, PROC_H);
    const ptsP = [AL, AR, BR, BL];

    // Terapkan efek
    const frame = offCtx.getImageData(0, 0, PROC_W, PROC_H);
    fxBuf.data.set(frame.data);
    effectFn(frame, fxBuf);

    // Mask poligon
    maskCtx.clearRect(0, 0, PROC_W, PROC_H);
    maskCtx.fillStyle = '#fff';
    maskCtx.beginPath();
    maskCtx.moveTo(ptsP[0].x, ptsP[0].y);
    for (let k = 1; k < ptsP.length; k++) maskCtx.lineTo(ptsP[k].x, ptsP[k].y);
    maskCtx.closePath();
    maskCtx.fill();
    const mask = maskCtx.getImageData(0, 0, PROC_W, PROC_H);

    composite(frame, fxBuf, mask);
    offCtx.putImageData(frame, 0, 0);

    // Simpan border untuk digambar setelah flush
    prism._borders.push({
      ptsD: [
        fingerPx(L, idxA, DW, DH),
        fingerPx(R, idxA, DW, DH),
        fingerPx(R, idxB, DW, DH),
        fingerPx(L, idxB, DW, DH),
      ],
      color,
    });
  }

  // Flush ke display
  ctx.drawImage(offCanvas, 0, 0, DW, DH);

  // Gambar border tiap persegi
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.lineJoin  = 'round';
  ctx.lineCap   = 'round';

  for (const { ptsD, color } of prism._borders) {
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 12;
    ctx.beginPath();
    ctx.moveTo(ptsD[0].x, ptsD[0].y);
    for (let k = 1; k < ptsD.length; k++) ctx.lineTo(ptsD[k].x, ptsD[k].y);
    ctx.closePath();
    ctx.stroke();
  }

  // Dots di tiap ujung jari
  const tips = [4, 8, 12, 16, 20];
  const dotColors = prism.sideEffects.map(e => BORDER_COLORS[e]);
  tips.forEach((idx, i) => {
    const lp = fingerPx(L, idx, DW, DH);
    const rp = fingerPx(R, idx, DW, DH);
    [lp, rp].forEach(p => {
      ctx.fillStyle   = dotColors[i];
      ctx.shadowColor = dotColors[i];
      ctx.shadowBlur  = 10;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  ctx.restore();
  prism._borders = [];
}
prism._borders = [];

// ── Render loop ───────────────────────────────────────────────────────────────
function loop() {
  animId = requestAnimationFrame(loop);
  if (video.readyState < 2 || !fxBuf) return;
  if (!handLandmarker) return; // hanya butuh hand model untuk jalan — face di-lazy load

  const DW = canvas.width, DH = canvas.height;
  const ts = performance.now();
  lastTimestamp = ts;

  // 1. Gambar video ke display (mirror)
  ctx.save();
  ctx.translate(DW, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, DW, DH);
  ctx.restore();

  // 2. Deteksi tangan
  const handResult = handLandmarker.detectForVideo(video, ts);
  const rawHands = {};
  if (handResult.landmarks && handResult.handedness) {
    handResult.landmarks.forEach((lm, i) => {
      rawHands[handResult.handedness[i][0].categoryName] = lm;
    });
  }
  
  // Smoothing landmarks untuk kurangi jitter
  const hands = smoothLandmarks(rawHands);

  const now = ts;

  // 3. Toggle box1: kedua tangan jepit telunjuk
  const bothIndex = hands.Left  && pinchDist(hands.Left,  8) < PINCH_THRESHOLD
                 && hands.Right && pinchDist(hands.Right, 8) < PINCH_THRESHOLD;
  if (bothIndex && !box1.triggered && now - box1.lastToggle > COOLDOWN_MS) {
    box1.active = !box1.active;
    box1.lastToggle = now;
  }
  box1.triggered = !!bothIndex;

  // 4. Toggle box2: kedua tangan jepit tengah
  const bothMiddle = hands.Left  && pinchDist(hands.Left,  12) < PINCH_THRESHOLD
                  && hands.Right && pinchDist(hands.Right, 12) < PINCH_THRESHOLD;
  if (bothMiddle && !box2.triggered && now - box2.lastToggle > COOLDOWN_MS) {
    box2.active = !box2.active;
    box2.lastToggle = now;
  }
  box2.triggered = !!bothMiddle;

  // 5. Cycle efek: satu tangan jepit kelingking
  const anyPinky  = (hands.Left  && pinchDist(hands.Left,  20) < PINCH_THRESHOLD)
                 || (hands.Right && pinchDist(hands.Right, 20) < PINCH_THRESHOLD);
  const bothPinky = hands.Left  && pinchDist(hands.Left,  20) < PINCH_THRESHOLD
                 && hands.Right && pinchDist(hands.Right, 20) < PINCH_THRESHOLD;
  const singlePinky = anyPinky && !bothPinky;
  
  if (singlePinky) {
    // Gesture aktif
    if (!cycleTriggered) {
      // Rising edge - gesture baru ditekan
      cycleHoldStart = now;
      cycleAutoMode = false;
      
      // Cycle sekali langsung
      if (now - cycleLastTime > COOLDOWN_MS) {
        if (box1.active) {
          box1.effectIdx = (box1.effectIdx + 1) % EFFECT_LIST.length;
        }
        if (box2.active) {
          box2.effectIdx = (box2.effectIdx + 1) % EFFECT_LIST.length;
        }
        cycleLastTime = now;
      }
      
      cycleTriggered = true;
    } else {
      // Gesture masih ditahan
      const holdDuration = now - cycleHoldStart;
      
      // Setelah 3 detik, masuk auto-cycle mode
      if (holdDuration >= 3000 && !cycleAutoMode) {
        cycleAutoMode = true;
        cycleLastTime = now;  // Reset untuk mulai auto-cycle
      }
      
      // Auto-cycle terus menerus saat ditahan
      if (cycleAutoMode && now - cycleLastTime > CYCLE_AUTO_INTERVAL) {
        if (box1.active) {
          box1.effectIdx = (box1.effectIdx + 1) % EFFECT_LIST.length;
        }
        if (box2.active) {
          box2.effectIdx = (box2.effectIdx + 1) % EFFECT_LIST.length;
        }
        cycleLastTime = now;
      }
    }
  } else {
    // Gesture dilepas - reset state
    cycleTriggered = false;
    cycleAutoMode = false;
  }

  // 5b. Toggle prisma: kedua tangan saling mendekatkan semua jari (prayer pose)
  // Jempol punya threshold lebih longgar karena anatominya sulit rapat
  const touchCount = (hands.Left && hands.Right) ? [
    { idx: 4,  thresh: 0.8 },  // jempol — paling longgar
    { idx: 8,  thresh: 0.4 },  // telunjuk
    { idx: 12, thresh: 0.4 },  // tengah
    { idx: 16, thresh: 0.4 },  // manis
    { idx: 20, thresh: 0.4 },  // kelingking
  ].filter(({ idx, thresh }) =>
    fingertipDist(hands.Left, hands.Right, idx) < thresh
  ).length : 0;
  const allTouch = touchCount >= 3;

  // Reset triggered saat tangan tidak terdeteksi agar rising edge bisa terjadi lagi
  if (!hands.Left || !hands.Right) prism.triggered = false;

  if (allTouch && !prism.triggered && now - prism.lastToggle > PRISM_COOLDOWN) {
    if (!prism.active) {
      // Assign 5 efek random berbeda untuk tiap persegi
      const shuffled = [...EFFECT_LIST].sort(() => Math.random() - 0.5);
      prism.sideEffects = shuffled.slice(0, 5);
      prism.active = true;
    } else {
      prism.active = false;
    }
    prism.lastToggle = now;
  }
  prism.triggered = !!allTouch;
  const boxAnyActive = box1.active || box2.active;
  const metalNow = (hands.Left  && isMetal(hands.Left))
                || (hands.Right && isMetal(hands.Right));
  if (!boxAnyActive && metalNow && !metalTriggered && now - metalLastTime > METAL_COOLDOWN) {
    if (!faceBlurActive) {
      // Aktifkan blur — pastikan face model sudah siap, load kalau belum
      ensureFaceLandmarker().then(ready => {
        if (ready || faceLandmarker) {
          faceBlurActive = true;
        }
      });
    } else {
      // Nonaktifkan blur
      faceBlurActive = false;
      lastFaceLandmarks = [];
    }
    metalLastTime = now;
  }
  metalTriggered = metalNow;

  // 7. Update landmark wajah setiap N frame
  faceFrameCount++;
  if (faceBlurActive && faceLandmarker && faceFrameCount % FACE_UPDATE_EVERY === 0) {
    const faceResult  = faceLandmarker.detectForVideo(video, ts);
    lastFaceLandmarks = faceResult.faceLandmarks || [];
  }
  if (!faceBlurActive) lastFaceLandmarks = [];

  // 8. Blur wajah dulu — hanya render saat tidak ada box/prisma aktif
  // Kalau ada box/prisma, blur akan di-composite dalam offscreen canvas
  if (faceBlurActive && lastFaceLandmarks.length > 0 && !box1.active && !box2.active && !prism.active) {
    blurFacesOval(lastFaceLandmarks, DW, DH);
  }

  // 9. Render efek persegi di atas blur wajah
  if (box1.active || box2.active) {
    // Hanya render kalau kedua tangan terdeteksi di frame ini
    const L = hands.Left;
    const R = hands.Right;

    if (L && R) {
      const sx = DW / PROC_W, sy = DH / PROC_H;

      // Gambar VIDEO mentah ke offscreen, lalu blur wajah akan di-composite via CPU
      offCtx.save();
      offCtx.translate(PROC_W, 0);
      offCtx.scale(-1, 1);
      offCtx.drawImage(video, 0, 0, PROC_W, PROC_H);
      offCtx.restore();

      // Jika blur wajah aktif, render ke offscreen dulu sebelum efek persegi
      if (faceBlurActive && lastFaceLandmarks.length > 0) {
        for (const landmarks of lastFaceLandmarks) {
          let minX = Infinity, maxX = -Infinity;
          let minY = Infinity, maxY = -Infinity;
          for (const lm of landmarks) {
            const x = (1 - lm.x) * PROC_W;
            const y = lm.y * PROC_H;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
          const rx = (maxX - minX) / 2 * 1.15, ry = (maxY - minY) / 2 * 1.10;

          offCtx.save();
          offCtx.beginPath();
          offCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
          offCtx.clip();
          offCtx.filter = "blur(20px)";
          offCtx.translate(PROC_W, 0);
          offCtx.scale(-1, 1);
          offCtx.drawImage(video, 0, 0, PROC_W, PROC_H);
          offCtx.restore();
        }
      }

      if (box1.active) {
        const key = EFFECT_LIST[box1.effectIdx];
        applyRegion(EFFECT_FNS[key], [lmPx(L,4), lmPx(R,4), lmPx(R,8), lmPx(L,8)],
          BORDER_COLORS[key], sx, sy, ts);
      }
      if (box2.active) {
        const key = EFFECT_LIST[box2.effectIdx];
        applyRegion(EFFECT_FNS[key], [lmPx(L,4), lmPx(R,4), lmPx(R,12), lmPx(L,12)],
          BORDER_COLORS[key], sx, sy, ts);
      }

      flushToDisplay(DW, DH);
    }
  }

  // 10. Render prisma segi lima (prioritas tertinggi — menimpa semua efek lain)
  if (prism.active) {
    if (hands.Left && hands.Right) {
      renderPrism(hands.Left, hands.Right, DW, DH);
    }
  }
  
  // 11. Visual feedback: glow di jari saat gesture aktif
  // Hanya render feedback saat ada tangan terdeteksi
  if (hands.Left || hands.Right) {
    // Index pinch (box1)
    if (bothIndex) {
      drawPinchGlow(hands.Left, 4, 8, '#00ff88', DW, DH);
      drawPinchGlow(hands.Right, 4, 8, '#00ff88', DW, DH);
    }
    
    // Middle pinch (box2)
    if (bothMiddle) {
      drawPinchGlow(hands.Left, 4, 12, '#ff00ff', DW, DH);
      drawPinchGlow(hands.Right, 4, 12, '#ff00ff', DW, DH);
    }
    
    // Pinky cycle
    if (singlePinky) {
      if (hands.Left && pinchDist(hands.Left, 20) < PINCH_THRESHOLD) {
        drawPinchGlow(hands.Left, 4, 20, '#ffd000', DW, DH);
      }
      if (hands.Right && pinchDist(hands.Right, 20) < PINCH_THRESHOLD) {
        drawPinchGlow(hands.Right, 4, 20, '#ffd000', DW, DH);
      }
    }
    
    // Metal gesture (face blur)
    if (metalNow) {
      if (hands.Left && isMetal(hands.Left)) {
        drawFingerGlow(hands.Left, 8, '#ff3366', DW, DH);  // index
        drawFingerGlow(hands.Left, 20, '#ff3366', DW, DH); // pinky
      }
      if (hands.Right && isMetal(hands.Right)) {
        drawFingerGlow(hands.Right, 8, '#ff3366', DW, DH);
        drawFingerGlow(hands.Right, 20, '#ff3366', DW, DH);
      }
    }
    
    // Prayer pose (prisma)
    if (allTouch && hands.Left && hands.Right) {
      const touchFingers = [4, 8, 12, 16, 20];
      for (const idx of touchFingers) {
        const dist = fingertipDist(hands.Left, hands.Right, idx);
        const thresh = idx === 4 ? 0.8 : 0.4;
        if (dist < thresh) {
          const color = '#00d4ff';
          drawFingerGlow(hands.Left, idx, color, DW, DH);
          drawFingerGlow(hands.Right, idx, color, DW, DH);
        }
      }
    }
  }
}
