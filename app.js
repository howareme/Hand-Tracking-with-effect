import {
  FilesetResolver,
  HandLandmarker,
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

// ── State ─────────────────────────────────────────────────────────────────────
const effects = {
  blur:    { active: false, triggered: false, lastToggle: 0 },
  thermal: { active: false, triggered: false, lastToggle: 0 },
  noir:    { active: false, triggered: false, lastToggle: 0 },
  glitch:  { active: false, triggered: false, lastToggle: 0 },
};

const PINCH_THRESHOLD = 0.15;
const COOLDOWN_MS     = 1000;

let landmarker = null;
let animId     = null;

// ── Buttons ───────────────────────────────────────────────────────────────────
btnStart.addEventListener("click", async () => {
  hero.classList.add("hidden");
  app.classList.remove("hidden");
  await initCamera();
  await initLandmarker();
  loadingOverlay.classList.add("done");
  loop();
});

btnBack.addEventListener("click", () => {
  if (animId) cancelAnimationFrame(animId);
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  Object.values(effects).forEach(e => { e.active = false; e.triggered = false; });
  loadingOverlay.classList.remove("done");
  app.classList.add("hidden");
  hero.classList.remove("hidden");
});

// ── Camera ────────────────────────────────────────────────────────────────────
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });
  await video.play();
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ── MediaPipe ─────────────────────────────────────────────────────────────────
async function initLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(vision, {
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
}

// ── Pinch ─────────────────────────────────────────────────────────────────────
function pinchDist(lm, fingerId) {
  const dx   = lm[4].x - lm[fingerId].x;
  const dy   = lm[4].y - lm[fingerId].y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const px   = lm[0].x - lm[5].x;
  const py   = lm[0].y - lm[5].y;
  const palm = Math.sqrt(px * px + py * py);
  return palm === 0 ? Infinity : dist / palm;
}

function tryToggle(key, bothPinching) {
  const e   = effects[key];
  const now = performance.now();
  if (bothPinching && !e.triggered && now - e.lastToggle > COOLDOWN_MS) {
    e.active     = !e.active;
    e.lastToggle = now;
  }
  e.triggered = bothPinching;
}

// ── Landmark → pixel coords ───────────────────────────────────────────────────
// Video digambar dengan flip horizontal (ctx.scale(-1,1)), sehingga koordinat X
// landmark harus di-mirror juga agar persegi sinkron dengan tampilan.
function lmPx(lm, idx, W, H) {
  return {
    x: (1 - lm[idx].x) * W,   // mirror X agar sesuai dengan video yang di-flip
    y: lm[idx].y * H,
  };
}

// ── Effects ───────────────────────────────────────────────────────────────────
function applyInvert(src, dst) {
  const s = src.data, d = dst.data;
  for (let i = 0; i < s.length; i += 4) {
    d[i]   = 255 - s[i];
    d[i+1] = 255 - s[i+1];
    d[i+2] = 255 - s[i+2];
    d[i+3] = 255;
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

let vigCache = null, vigW = 0, vigH = 0;
function getVignette(w, h) {
  if (vigCache && vigW === w && vigH === h) return vigCache;
  vigCache = new Float32Array(w*h); vigW = w; vigH = h;
  const cx = w/2, cy = h/2;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      vigCache[y*w+x] = Math.max(0.2, Math.min(1.0, 1.0 - Math.sqrt(((x-cx)/cx)**2+((y-cy)/cy)**2)*0.8));
  return vigCache;
}

function applyNoir(src, dst, w, h) {
  const s = src.data, d = dst.data, vign = getVignette(w, h);
  for (let i = 0; i < s.length; i += 4) {
    const c = Math.min(255, Math.max(0, (s[i]*0.299+s[i+1]*0.587+s[i+2]*0.114 - 60)*1.6)) * vign[i>>2];
    d[i] = d[i+1] = d[i+2] = c; d[i+3] = 255;
  }
}

function applyGlitch(src, dst, w, h) {
  const s = src.data, d = dst.data, shift = 8;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i  = (y*w+x)*4;
      const iR = (y*w+Math.min(w-1,x+shift))*4;
      const iB = (y*w+Math.max(0,x-shift))*4;
      d[i] = s[iR]; d[i+1] = s[i+1]; d[i+2] = s[iB+2]; d[i+3] = 255;
    }
  }
  const seed = Math.floor(performance.now()/80);
  for (let n = 0; n < 6; n++) {
    const y   = Math.abs(Math.sin(seed*7.3+n*13.7)*h)|0;
    const off = (Math.sin(seed*3.1+n*5.9)*20)|0;
    const row = new Uint8Array(w*4);
    for (let x = 0; x < w; x++) {
      const s2 = (y*w+Math.min(w-1,Math.max(0,x+off)))*4;
      row[x*4] = d[s2]; row[x*4+1] = d[s2+1]; row[x*4+2] = d[s2+2]; row[x*4+3] = 255;
    }
    dst.data.set(row, y*w*4);
  }
}

// ── Mask + composite ──────────────────────────────────────────────────────────
function buildMask(pts, w, h) {
  const mc = document.createElement("canvas");
  mc.width = w; mc.height = h;
  const mx = mc.getContext("2d");
  mx.fillStyle = "#fff";
  mx.beginPath();
  mx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) mx.lineTo(pts[i].x, pts[i].y);
  mx.closePath();
  mx.fill();
  return mx.getImageData(0, 0, w, h);
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

function drawBorder(pts, colors) {
  for (const [color, lw] of colors) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────
function loop() {
  animId = requestAnimationFrame(loop);
  if (video.readyState < 2) return;

  const W = canvas.width, H = canvas.height;

  // Gambar video dengan flip horizontal — mirror seperti kaca
  ctx.save();
  ctx.translate(W, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

  // Deteksi tangan dari video asli (sebelum flip)
  const result = landmarker.detectForVideo(video, performance.now());

  const hands = {};
  if (result.landmarks && result.handedness) {
    result.landmarks.forEach((lm, i) => {
      hands[result.handedness[i][0].categoryName] = lm;
    });
  }

  // Toggle efek
  const fingerMap = { blur: 8, thermal: 12, noir: 16, glitch: 20 };
  for (const [key, fid] of Object.entries(fingerMap)) {
    const both = hands.Left  && pinchDist(hands.Left,  fid) < PINCH_THRESHOLD
              && hands.Right && pinchDist(hands.Right, fid) < PINCH_THRESHOLD;
    tryToggle(key, both);
  }

  if (!hands.Left || !hands.Right) return;

  const L = hands.Left, R = hands.Right;

  // Karena video sudah di-flip saat draw, lmPx juga flip X agar persegi sinkron
  const base = ctx.getImageData(0, 0, W, H);

  const applyRegion = (imgData, effectFn, pts, borderColors) => {
    const fx   = new ImageData(new Uint8ClampedArray(imgData.data), W, H);
    effectFn(imgData, fx, W, H);
    const mask = buildMask(pts, W, H);
    composite(imgData, fx, mask);
    ctx.putImageData(imgData, 0, 0);
    drawBorder(pts, borderColors);
  };

  if (effects.blur.active) {
    const pts = [lmPx(L,4,W,H), lmPx(R,4,W,H), lmPx(R,8,W,H), lmPx(L,8,W,H)];
    applyRegion(ctx.getImageData(0,0,W,H), applyInvert, pts,
      [["rgba(255,255,255,0.6)",4],["rgba(180,180,180,0.4)",2]]);
  }

  if (effects.thermal.active) {
    const pts = [lmPx(L,4,W,H), lmPx(R,4,W,H), lmPx(R,12,W,H), lmPx(L,12,W,H)];
    applyRegion(ctx.getImageData(0,0,W,H), applyThermal, pts,
      [["rgba(255,112,67,0.8)",6],["rgba(255,220,0,0.6)",2]]);
  }

  if (effects.noir.active) {
    const pts = [lmPx(L,4,W,H), lmPx(R,4,W,H), lmPx(R,16,W,H), lmPx(L,16,W,H)];
    applyRegion(ctx.getImageData(0,0,W,H), applyNoir, pts,
      [["rgba(50,50,50,0.9)",6],["rgba(200,200,200,0.6)",2]]);
  }

  if (effects.glitch.active) {
    const pts = [lmPx(L,4,W,H), lmPx(R,4,W,H), lmPx(R,20,W,H), lmPx(L,20,W,H)];
    applyRegion(ctx.getImageData(0,0,W,H), applyGlitch, pts,
      [["rgba(0,255,0,0.7)",5],["rgba(255,0,200,0.6)",2]]);
  }
}
