"use strict";

const $ = (id) => document.getElementById(id);
const fields = ["pipeline","detScore","fallback","modelSource","modelUrl","normalization","backend","region","deviceId","mqttTopic","subscribeTopic","brokerUrl","threshold","stableFrames","cooldown","mqttUsername"];
const STORAGE_KEY = "prog6002-classifier", LOG_VISIBLE_KEY = "prog6002-log-visible";
let model = null, stream = null, running = false, mqttClient = null, subscribedTopic = "";
let candidate = "", candidateFrames = 0, lastPublishedClass = "", lastPublishedAt = 0;
let sequence = 0, published = 0, received = 0, errorCount = 0, animationId = null, inferenceErrors = 0;

// ---------- Event / error log ----------
function log(message, level="info") {
  const item = document.createElement("li");
  item.className = level;
  item.textContent = `${new Date().toLocaleTimeString()} · ${message}`;
  $("eventLog").prepend(item);
  while ($("eventLog").children.length > 100) $("eventLog").lastChild.remove();
  if (level === "error") { errorCount += 1; updateErrorBadge(); }
}

function updateErrorBadge() {
  $("errorBadge").textContent = errorCount;
  $("errorBadge").hidden = errorCount === 0;
}

function setLogVisible(visible) {
  $("logPanel").hidden = !visible;
  $("logToggle").setAttribute("aria-expanded", String(visible));
  $("logToggle").firstChild.textContent = visible ? "Hide log " : "Show log ";
  try { localStorage.setItem(LOG_VISIBLE_KEY, visible ? "1" : "0"); } catch {}
}

function clearLog() {
  $("eventLog").replaceChildren();
  errorCount = 0; updateErrorBadge();
}

function setText(id, value) { $(id).textContent = value; }
function setOverall(text, kind="warn") { const el=$("overallStatus"); el.textContent=text; el.className=`pill ${kind}`; }
function value(id) { return $(id).value.trim(); }

// ---------- Settings ----------
function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    fields.forEach(id => { if (saved[id] !== undefined) $(id).value = saved[id]; });
  } catch (error) { log(`Saved configuration ignored: ${error.message}`, "error"); }
  let logVisible = true;
  try { logVisible = localStorage.getItem(LOG_VISIBLE_KEY) !== "0"; } catch {}
  setLogVisible(logVisible);
}

function saveSettings() {
  try {
    const data = Object.fromEntries(fields.map(id => [id, $(id).value]));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    log("Configuration saved on this tablet (password is not stored).");
  } catch (error) { log(`Could not save configuration: ${error.message}`, "error"); }
}

// ---------- Model URL ----------
function normalizedModelUrl() {
  const raw = value("modelUrl");
  if (!/^https:\/\//i.test(raw)) throw new Error("Model URL must start with https://");
  return raw.endsWith("/") ? raw : raw + "/";
}

// ---------- Model loading ----------
// Every model is wrapped so the rest of the app can call
// predict(videoElement) -> [{className, probability}] and getTotalClasses().

async function loadTeachableMachineUrl() {
  const base = normalizedModelUrl();
  const tm = await tmImage.load(base + "model.json", base + "metadata.json");
  // Teachable Machine does its own centre-crop; region and preview are not supported.
  return { predict: async (src) => ({ predictions: await tm.predict(src, false), stats: null }),
           getTotalClasses: () => tm.getTotalClasses(),
           dispose: () => tm.dispose?.(), source: base,
           description: `Teachable Machine model with ${tm.getTotalClasses()} classes` };
}

function readFileText(file) { return file.text(); }

async function readLabels(files) {
  const txt = files.find(f => /\.txt$/i.test(f.name));
  if (txt) return (await readFileText(txt)).split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const meta = files.find(f => /metadata.*\.json$/i.test(f.name));
  if (meta) {
    const data = JSON.parse(await readFileText(meta));
    if (Array.isArray(data.labels)) return data.labels;
  }
  return [];
}

async function loadModelFiles() {
  const files = [...$("modelFiles").files];
  if (!files.length) throw new Error("Select the model files first (model.json + .bin weights).");
  const jsonFiles = files.filter(f => /\.json$/i.test(f.name) && !/metadata/i.test(f.name));
  if (jsonFiles.length !== 1) throw new Error(`Select exactly one model JSON file (found ${jsonFiles.length}).`);
  const modelJsonFile = jsonFiles[0];
  const modelJson = JSON.parse(await readFileText(modelJsonFile));
  if (!modelJson.modelTopology || !modelJson.weightsManifest) throw new Error(`${modelJsonFile.name} is not a TensorFlow.js model file.`);

  // Check every weight shard named in the manifest was selected.
  const weightFiles = files.filter(f => /\.bin$/i.test(f.name));
  const needed = modelJson.weightsManifest.flatMap(group => group.paths).map(p => p.split("/").pop());
  const missing = needed.filter(name => !weightFiles.some(f => f.name === name));
  if (missing.length) throw new Error(`Missing weight file(s): ${missing.join(", ")}`);

  const handler = tf.io.browserFiles([modelJsonFile, ...weightFiles]);
  const isGraph = modelJson.format === "graph-model";
  const net = isGraph ? await tf.loadGraphModel(handler) : await tf.loadLayersModel(handler);
  return wrapTfjsModel(net, {
    labels: await readLabels(files), normalization: $("normalization").value,
    source: `local:${modelJsonFile.name}`, name: `${isGraph ? "Graph" : "Layers"} model from ${modelJsonFile.name}`
  });
}

// Waste models shipped with the app. Each folder has model.json, weights and metadata.json.
const BUNDLED_MODELS = {
  default: { dir: "models/ecovision-mobilenetv3/", title: "EcoVision MobileNetV3" },   // tools/convert_ecovision.py
  lite:    { dir: "models/waste-mobilenetv2/",     title: "Lite TrashNet MobileNetV2" } // tools/train_waste_model.py
};

async function loadBundledModel(key) {
  const {dir, title} = BUNDLED_MODELS[key];
  if (location.protocol === "file:") throw new Error("Bundled models need the page served over http(s), not opened as a file.");
  const metaResponse = await fetch(dir + "metadata.json");
  if (!metaResponse.ok) throw new Error(`Bundled model metadata not found (${metaResponse.status}).`);
  const meta = await metaResponse.json();
  const net = meta.format === "graph-model" ? await tf.loadGraphModel(dir + "model.json")
                                            : await tf.loadLayersModel(dir + "model.json");
  return wrapTfjsModel(net, {
    labels: meta.labels ?? [], normalization: meta.normalization ?? "-1to1",
    source: `bundled:${meta.modelName ?? key}`, name: `${title} (${meta.architecture ?? "bundled"})`
  });
}

// ---------- Object detection (stage 1 of the hybrid pipeline) ----------
// COCO-SSD finds everyday objects (80 COCO classes); the largest one is cropped and passed to the
// material classifier. This keeps the background out of the classifier and gives a natural "no item".
const DETECTOR_URL = "models/coco-ssd-lite/model.json";
// COCO classes that are never the waste item being shown: people (hands) and background furniture/vehicles.
const IGNORED_OBJECTS = new Set(["person", "dining table", "chair", "couch", "bed", "toilet", "tv", "refrigerator",
  "oven", "sink", "bench", "potted plant", "car", "bus", "truck", "train", "airplane", "boat", "motorcycle", "bicycle"]);
let detector = null;
const cropCanvas = document.createElement("canvas");

function usesDetection() { return $("pipeline").value === "hybrid"; }

async function loadDetector() {
  if (typeof cocoSsd === "undefined") throw new Error("COCO-SSD library failed to load. Check the internet connection and reload.");
  await ensureBackend();
  setText("detectedStatus", "Loading detector…");
  detector = await cocoSsd.load({ base: "lite_mobilenet_v2", modelUrl: DETECTOR_URL });
  const warmup = tf.zeros([300, 300, 3], "int32");
  await detector.detect(warmup); warmup.dispose();
  setText("detectedStatus", "—");
  log("Object detector loaded: COCO-SSD lite MobileNetV2 (80 everyday object classes).");
}

function disposeDetector() { detector?.dispose(); detector = null; }

function sourceSize(src) {
  return src instanceof HTMLVideoElement ? [src.videoWidth, src.videoHeight]
       : src instanceof HTMLImageElement ? [src.naturalWidth, src.naturalHeight] : [src.width, src.height];
}

// Square crop around a detection box with 15% padding, kept inside the frame.
function squareAround([x, y, w, h], frameW, frameH) {
  const side = Math.min(Math.max(w, h) * 1.15, frameW, frameH);
  const cx = x + w / 2, cy = y + h / 2;
  return { x: Math.min(Math.max(cx - side / 2, 0), frameW - side), y: Math.min(Math.max(cy - side / 2, 0), frameH - side), w: side, h: side };
}

// Detect -> crop -> classify. Returns {predictions, stats, detection, detections, mode}.
// mode: "classify" (no detector), "detect" (object found), "fallback" (none found, centre classified), "none".
async function analyseFrame(src) {
  const region = $("region").value, preview = $("inputPreview");
  // The video can briefly have no picture (starting up, phone rotating); skip such frames.
  const [fw0, fh0] = sourceSize(src);
  if (!fw0 || !fh0 || (src instanceof HTMLVideoElement && src.readyState < 2)) return { predictions: null, detections: [], mode: "waiting" };
  if (!usesDetection()) return { ...(await model.predict(src, {region, preview})), detection: null, detections: [], mode: "classify" };
  const detections = (await detector.detect(src, 10, Number(value("detScore")) || 0.4))
    .filter(d => !IGNORED_OBJECTS.has(d.class));
  if (!detections.length) {
    if ($("fallback").value === "centre") return { ...(await model.predict(src, {region, preview})), detection: null, detections, mode: "fallback" };
    return { predictions: null, stats: null, detection: null, detections, mode: "none" };
  }
  const detection = detections.reduce((a, b) => b.bbox[2] * b.bbox[3] > a.bbox[2] * a.bbox[3] ? b : a);
  const [fw, fh] = sourceSize(src), r = squareAround(detection.bbox, fw, fh);
  cropCanvas.width = cropCanvas.height = 256;
  cropCanvas.getContext("2d").drawImage(src, r.x, r.y, r.w, r.h, 0, 0, 256, 256);
  return { ...(await model.predict(cropCanvas, {region: "full", preview})), detection, detections, mode: "detect" };
}

// Maps a rectangle in camera-frame pixels to the displayed video (object-fit: contain).
function videoToScreen(video) {
  const vw = video.videoWidth, vh = video.videoHeight, cw = video.clientWidth, ch = video.clientHeight;
  const scale = Math.min(cw / vw, ch / vh);
  return { scale, ox: (cw - vw * scale) / 2, oy: (ch - vh * scale) / 2 };
}

function drawDetections(detections = [], primary = null, material = null) {
  const video = $("camera"), canvas = $("detectionOverlay");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = video.clientWidth * dpr; canvas.height = video.clientHeight * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!video.videoWidth || !detections.length) return;
  const {scale, ox, oy} = videoToScreen(video);
  ctx.font = "600 13px system-ui, sans-serif"; ctx.textBaseline = "top";
  for (const d of detections) {
    const [x, y, w, h] = d.bbox.map(v => v * scale), isPrimary = d === primary;
    ctx.lineWidth = isPrimary ? 3 : 1.5;
    ctx.strokeStyle = isPrimary ? "#19e28a" : "#ffffffaa";
    ctx.strokeRect(ox + x, oy + y, w, h);
    const text = `${d.class} ${(d.score * 100).toFixed(0)}%` +
                 (isPrimary && material ? ` → ${material.className} ${(material.probability * 100).toFixed(0)}%` : "");
    const tw = ctx.measureText(text).width + 10, ty = Math.max(oy + y - 20, 0);
    ctx.fillStyle = isPrimary ? "#0d6b44ee" : "#071723cc";
    ctx.fillRect(ox + x, ty, tw, 19);
    ctx.fillStyle = "white"; ctx.fillText(text, ox + x + 5, ty + 3);
  }
}

function describeDetection({mode, detection, detections}) {
  if (mode === "classify") return "Not used";
  if (mode === "waiting") return "Waiting for camera…";
  if (mode === "detect") return `${detection.class} ${(detection.score * 100).toFixed(0)}%` + (detections.length > 1 ? ` (+${detections.length - 1} more)` : "");
  return mode === "fallback" ? "Nothing (classified centre)" : "Nothing detected";
}

// ---------- Compute backend ----------
async function ensureBackend() {
  const wanted = $("backend").value;
  if (wanted === "wasm" && tf.wasm?.setWasmPaths) tf.wasm.setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/");
  if (tf.getBackend() !== wanted) {
    const ok = await tf.setBackend(wanted).catch(() => false);
    if (!ok) { log(`Could not start the ${wanted} backend; using ${tf.getBackend()}.`, "error"); }
  }
  await tf.ready();
  const info = backendInfo();
  setText("backendStatus", info);
  return info;
}

function backendInfo() {
  const name = tf.getBackend();
  if (name !== "webgl") return name === "wasm" ? "WebAssembly (CPU)" : name;
  let gpu = "unknown GPU";
  try {
    const gl = tf.backend().gpgpu.gl, ext = gl.getExtension("WEBGL_debug_renderer_info");
    gpu = gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
  } catch {}
  const bits = tf.env().getBool("WEBGL_RENDER_FLOAT32_CAPABLE") ? "32-bit" : "16-bit";
  return `WebGL ${tf.env().getNumber("WEBGL_VERSION")} (${gpu}, ${bits} float)`;
}

async function backendChanged() {
  if (running) { log("Stop classification before changing the compute backend.", "error"); return; }
  modelSettingsChanged(); disposeDetector();
  try { log(`Compute backend: ${await ensureBackend()}.`); } catch (error) { log(error.message, "error"); }
}

// ---------- Diagnostics ----------
const SELFTEST_DIR = "models/selftest/";
let blankFrames = 0, lastBlankWarning = 0;

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load image ${url}`));
    img.src = url;
  });
}

function showInputStats(stats) {
  if (!stats) { setText("inputStats", "Preview not available for Teachable Machine URL models."); return; }
  setText("inputStats", `Brightness ${stats.brightness.toFixed(0)} / 255 · Contrast ${stats.contrast.toFixed(0)}`);
  blankFrames = stats.contrast < 8 ? blankFrames + 1 : 0;
  if (blankFrames >= 8 && Date.now() - lastBlankWarning > 10000) {
    lastBlankWarning = Date.now();
    log(`Camera frames look blank (brightness ${stats.brightness.toFixed(0)}, contrast ${stats.contrast.toFixed(1)}). ` +
        "Blank input makes the models answer cardboard/paper. Check the preview, lighting and lens.", "warn");
  }
}

// Runs the bundled model on reference images and compares with results from a known-good backend.
async function runSelfTest() {
  const key = $("modelSource").value;
  if (!BUNDLED_MODELS[key]) { log("Self-test is available for the bundled models only.", "warn"); return; }
  $("selfTestButton").disabled = true;
  try {
    await loadPipeline();
    const all = await (await fetch(SELFTEST_DIR + "expected.json")).json(), expected = all[key];
    let passed = 0, total = Object.keys(expected).length;
    if (detector) {
      for (const [file, ref] of Object.entries(all.detector)) {
        const found = await detector.detect(await loadImage(SELFTEST_DIR + file), 10, 0.2);
        const best = found.length ? found.reduce((a, b) => b.bbox[2] * b.bbox[3] > a.bbox[2] * a.bbox[3] ? b : a) : null;
        const ok = best?.class === ref.label && Math.abs(best.score - ref.score) < 0.1;
        passed += ok; total += 1;
        log(`Self-test detector ${file}: got ${best ? `${best.class} ${(best.score * 100).toFixed(1)}%` : "nothing"}, ` +
            `reference ${ref.label} ${(ref.score * 100).toFixed(1)}% → ${ok ? "PASS" : "FAIL"}`, ok ? "info" : "error");
      }
    }
    for (const [file, ref] of Object.entries(expected)) {
      const {predictions} = await model.predict(await loadImage(SELFTEST_DIR + file), {region: "full", preview: $("inputPreview")});
      const top = predictions.reduce((a, b) => b.probability > a.probability ? b : a);
      // Compare every class probability with the reference (robust even when two classes are close).
      const diff = Math.max(...predictions.map(p => Math.abs(p.probability - (ref.probabilities[p.className] ?? 0))));
      const ok = diff < 0.1;
      passed += ok;
      log(`Self-test ${file}: got ${top.className} ${(top.probability * 100).toFixed(1)}%, reference ${ref.label}; ` +
          `largest difference ${(diff * 100).toFixed(1)} points → ${ok ? "PASS" : "FAIL"}`, ok ? "info" : "error");
    }
    const info = backendInfo();
    if (passed === total) log(`Self-test passed ${passed}/${total} on ${info}. The model computes correctly on this device.`);
    else log(`Self-test FAILED ${total - passed}/${total} on ${info}. This backend gives wrong results on this device; ` +
             "choose WebAssembly under Compute backend and re-test.", "error");
  } catch (error) { log(`Self-test error: ${error.message}`, "error"); }
  finally { $("selfTestButton").disabled = false; }
}

// Classifies a still photo (from the gallery or the phone's camera app) without the live video path.
async function classifyPhoto() {
  const file = $("photoInput").files[0];
  if (!file) return;
  try {
    await loadPipeline();
    const url = URL.createObjectURL(file);
    const img = await loadImage(url);
    URL.revokeObjectURL(url);
    // Downscale large phone photos before classification.
    const scale = Math.min(1, 800 / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale); canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    const frame = await analyseFrame(canvas);
    setText("detectedStatus", describeDetection(frame));
    if (!frame.predictions) { showNoItem(); log(`Photo ${file.name}: no object detected.`, "warn"); return; }
    const {sorted} = showPredictions(frame.predictions);
    showInputStats(frame.stats);
    const found = frame.detection ? `detected ${frame.detection.class} ${(frame.detection.score * 100).toFixed(0)}% → ` : "";
    log(`Photo ${file.name}: ${found}` + sorted.slice(0, 3).map(p => `${p.className} ${(p.probability * 100).toFixed(1)}%`).join(", "));
  } catch (error) { log(`Photo classification failed: ${error.message}`, "error"); }
  finally { $("photoInput").value = ""; }
}

// Draws the model's region on top of the video so users can frame the item.
function updateRegionGuide() {
  const video = $("camera"), guide = $("regionGuide");
  const vw = video.videoWidth, vh = video.videoHeight;
  // In detection mode the region is only used when nothing is detected and the fallback is "centre".
  const regionUsed = !usesDetection() || $("fallback").value === "centre";
  if (!vw || !vh || !regionUsed || $("region").value === "full") { guide.hidden = true; return; }
  const {scale, ox, oy} = videoToScreen(video);
  const r = regionRect(vw, vh, $("region").value);
  Object.assign(guide.style, { left: `${ox + r.x * scale}px`, top: `${oy + r.y * scale}px`,
                               width: `${r.w * scale}px`, height: `${r.h * scale}px` });
  guide.hidden = false;
}

// Region of the camera frame sent to the model: "full" frame, or a centred square with optional zoom.
function regionRect(width, height, region) {
  if (region === "full") return { x: 0, y: 0, w: width, h: height };
  const zoom = { zoom15: 1.5, zoom2: 2 }[region] ?? 1;
  const size = Math.round(Math.min(width, height) / zoom);
  return { x: Math.floor((width - size) / 2), y: Math.floor((height - size) / 2), w: size, h: size };
}

// Wraps any TF.js image classifier in the app's interface:
// predict(source, {region, preview}) -> {predictions: [{className, probability}], stats: {brightness, contrast}}
// source can be a <video>, <img> or <canvas>; preview is an optional canvas showing exactly what the model sees.
function wrapTfjsModel(net, {labels, normalization, source, name}) {
  const inputShape = net.inputs[0].shape;          // e.g. [null, 224, 224, 3]
  const height = inputShape[1] > 0 ? inputShape[1] : 224;
  const width = inputShape[2] > 0 ? inputShape[2] : 224;
  const channels = inputShape[3] > 0 ? inputShape[3] : 3;
  const outputSize = net.outputs[0].shape?.at(-1);

  if (outputSize > 0 && labels.length !== outputSize) {
    if (labels.length) log(`Label count (${labels.length}) does not match model outputs (${outputSize}); using generic names for extras.`, "warn");
    labels = Array.from({length: outputSize}, (_, i) => labels[i] ?? `Class ${i + 1}`);
  }

  // Crop the chosen region and resize to the model's input size. Result: [H, W, 3] float, 0-255.
  function cropResize(source, region) {
    return tf.tidy(() => {
      let img = tf.browser.fromPixels(source);       // [H, W, 3] int32
      const r = regionRect(img.shape[1], img.shape[0], region);
      if (r.w !== img.shape[1] || r.h !== img.shape[0]) img = img.slice([r.y, r.x, 0], [r.h, r.w, 3]);
      return tf.image.resizeBilinear(img, [height, width]).toFloat();
    });
  }

  function normalise(view) {
    return tf.tidy(() => {
      let img = channels === 1 ? view.mean(2, true) : view;
      if (normalization === "-1to1") img = img.div(127.5).sub(1);
      else if (normalization === "0to1") img = img.div(255);
      return img.expandDims(0);
    });
  }

  async function predict(src, {region = "square", preview = null} = {}) {
    const view = cropResize(src, region);
    try {
      const scores = tf.tidy(() => {
        let out = net.predict(normalise(view));
        if (Array.isArray(out)) out = out[0];
        out = out.squeeze();
        // Apply softmax if the model outputs logits rather than probabilities.
        const sum = out.sum().dataSync()[0], min = out.min().dataSync()[0];
        return (min < 0 || Math.abs(sum - 1) > 0.01) ? tf.softmax(out) : out;
      });
      const values = await scores.data();
      scores.dispose();
      // Brightness = mean pixel value; contrast = standard deviation. Near-zero contrast means a blank frame.
      const stats = tf.tidy(() => { const {mean, variance} = tf.moments(view);
        return { brightness: mean.dataSync()[0], contrast: Math.sqrt(variance.dataSync()[0]) }; });
      if (preview) {
        const pixels = tf.tidy(() => view.div(255).clipByValue(0, 1));
        await tf.browser.toPixels(pixels, preview);
        pixels.dispose();
      }
      return { stats, predictions: Array.from(values, (p, i) => ({ className: labels[i] ?? `Class ${i + 1}`, probability: p })) };
    } finally { view.dispose(); }
  }

  // Warm up once so the first real frame is not slow.
  tf.tidy(() => { net.predict(tf.zeros([1, height, width, channels])); });

  return { predict, getTotalClasses: () => labels.length, dispose: () => net.dispose(), source,
           description: `${name}: input ${width}×${height}×${channels}, ${labels.length} classes ` +
                        `(${labels.join(", ")}), ${normalization} normalisation` };
}

async function loadModel() {
  if (running) throw new Error("Stop classification before loading a different model.");
  if (typeof tf === "undefined") throw new Error("TensorFlow.js failed to load. Check the internet connection and reload.");
  const source = $("modelSource").value;
  setText("modelStatus", "Loading…");
  try {
    await ensureBackend();
    const loaded = source === "files" ? await loadModelFiles()
                 : source === "url" ? await loadTeachableMachineUrl()
                 : await loadBundledModel(BUNDLED_MODELS[source] ? source : "default");
    if (model) model.dispose?.();
    model = loaded;
  } catch (error) {
    setText("modelStatus", model ? "Load failed (previous model kept)" : "Load failed");
    throw new Error(`Model load failed: ${error.message}`);
  }
  setText("modelStatus", `${model.getTotalClasses()} classes loaded`);
  log(`Model loaded: ${model.description}.`);
  $("className").textContent = "—"; $("predictions").replaceChildren();
}

// Loads whatever the chosen pipeline needs: the material classifier, plus the detector in hybrid mode.
async function loadPipeline() {
  if (!model) await loadModel();
  if (usesDetection() && !detector) {
    try { await loadDetector(); }
    catch (error) { setText("detectedStatus", "Detector failed"); throw new Error(`Detector load failed: ${error.message}`); }
  }
}

async function loadModelButton() {
  $("loadModelButton").disabled = true;
  try { await loadPipeline(); } catch (error) { log(error.message, "error"); }
  finally { $("loadModelButton").disabled = false; }
}

function pipelineChanged() {
  $("detectionFields").hidden = !usesDetection();
  if (!usesDetection()) { drawDetections(); setText("detectedStatus", "Not used"); }
  else if (!detector) setText("detectedStatus", "—");
  updateRegionGuide();
}

function updateModelSourceUi() {
  const source = $("modelSource").value;
  $("defaultSourceFields").hidden = source !== "default";
  $("liteSourceFields").hidden = source !== "lite";
  $("urlSourceFields").hidden = source !== "url";
  $("fileSourceFields").hidden = source !== "files";
}

function showSelectedFiles() {
  const files = [...$("modelFiles").files];
  $("modelFileList").replaceChildren(...files.map(f => {
    const li = document.createElement("li");
    li.textContent = `${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
    return li;
  }));
  if (files.length) log(`Selected ${files.length} model file(s). Tap “Load model” or start the classifier.`);
}

function modelSettingsChanged() {
  if (!model || running) return;
  model.dispose?.(); model = null;
  setText("modelStatus", "Not loaded");
  log("Model settings changed; the model will be reloaded on next start.");
}

// ---------- Camera ----------
function cameraErrorMessage(error) {
  switch (error.name) {
    case "NotAllowedError": return "Camera permission denied. Allow camera access in the browser site settings.";
    case "NotFoundError": return "No camera found on this device.";
    case "NotReadableError": return "Camera is in use by another app or tab.";
    case "OverconstrainedError": return "Camera does not support the requested settings.";
    default: return `Camera error: ${error.message}`;
  }
}

async function startCamera() {
  if (stream) return;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable. Open this page over HTTPS in a current browser.");
  setText("cameraStatus", "Starting…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio:false, video:{facingMode:{ideal:"environment"}, width:{ideal:640}, height:{ideal:480}}
    });
  } catch (error) {
    setText("cameraStatus", "Failed");
    throw new Error(cameraErrorMessage(error));
  }
  $("camera").srcObject = stream;
  await $("camera").play();
  $("cameraMessage").classList.add("hidden");
  updateRegionGuide();
  const settings = stream.getVideoTracks()[0]?.getSettings() || {};
  setText("cameraStatus", `Running${settings.width ? ` (${settings.width}×${settings.height})` : ""}`);
  log(`Camera started${settings.width ? ` at ${settings.width}×${settings.height}` : ""}.`);
  $("stopButton").disabled = false;
}

async function testCamera() {
  if (stream) { log("Camera is already running."); return; }
  $("cameraTestButton").disabled = true;
  try {
    await startCamera();
    setOverall("Camera test", "ok");
  } catch (error) { log(error.message, "error"); setOverall("Camera failed", "bad"); }
  finally { $("cameraTestButton").disabled = false; }
}

function stopAll() {
  const wasActive = Boolean(stream) || running;
  running = false;
  if (animationId) cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  $("camera").srcObject = null;
  $("regionGuide").hidden = true;
  drawDetections();
  $("cameraMessage").textContent = "Camera stopped";
  $("cameraMessage").classList.remove("hidden");
  $("startButton").disabled = false; $("stopButton").disabled = true;
  setText("cameraStatus", "Stopped");
  setOverall("Stopped", "warn");
  if (wasActive) log("Camera stopped.");
}

// ---------- Classification ----------
function showPredictions(predictions) {
  const sorted = [...predictions].sort((a,b)=>b.probability-a.probability);
  const top = sorted[0];
  setText("className", top.className);
  setText("confidenceText", `${(top.probability*100).toFixed(1)}%`);
  $("confidenceBar").style.width = `${top.probability*100}%`;
  $("predictions").replaceChildren(...sorted.slice(0,5).map(p => {
    const row=document.createElement("div"); row.className="prediction";
    const a=document.createElement("span"), b=document.createElement("span");
    a.textContent=p.className; b.textContent=`${(p.probability*100).toFixed(1)}%`; row.append(a,b); return row;
  }));
  return {top, sorted};
}

// Nothing detected: show "No item" and restart the stable-frame count, so nothing is published.
function showNoItem() {
  setText("className", "No item"); setText("confidenceText", "—");
  $("confidenceBar").style.width = "0"; $("predictions").replaceChildren();
  candidate = ""; candidateFrames = 0;
}

function updateStability(label) {
  if (label === candidate) candidateFrames += 1;
  else { candidate = label; candidateFrames = 1; }
}

// frame = result of analyseFrame (null for manual tests). The bounding box is normalised to 0-1 of the frame.
function buildPayload(top, sorted, inferenceMs, source="camera", frame=null) {
  const modelUrl = model?.source ?? null;
  const d = frame?.detection, [fw, fh] = d ? sourceSize($("camera")) : [1, 1];
  return {
    schema_version:1, message_type:"waste_classification", device_id:value("deviceId"),
    sequence:++sequence, timestamp:new Date().toISOString(), source,
    pipeline: frame?.mode ?? null,
    classification:top.className, confidence:Number(top.probability.toFixed(4)),
    detected_object: d ? { label:d.class, confidence:Number(d.score.toFixed(4)),
      bbox: [d.bbox[0]/fw, d.bbox[1]/fh, d.bbox[2]/fw, d.bbox[3]/fh].map(v => Number(v.toFixed(4))) } : null,
    inference_ms:Math.round(inferenceMs), model_url:modelUrl,
    alternatives:sorted.slice(1,3).map(p=>({label:p.className, confidence:Number(p.probability.toFixed(4))}))
  };
}

function publishText(text, description) {
  if (!mqttClient?.connected) { log("Cannot publish: MQTT is disconnected.", "error"); return false; }
  const topic=value("mqttTopic");
  if (!topic) { log("Cannot publish: publish topic is empty.", "error"); return false; }
  mqttClient.publish(topic, text, {qos:1, retain:false}, error => {
    if (error) return log(`Publish failed: ${error.message}`, "error");
    published += 1; setText("publishedCount", published); setText("payloadPreview", prettyPrint(text));
    log(`Published ${description} to ${topic}.`);
  });
  return true;
}

function publishPayload(payload) {
  return publishText(JSON.stringify(payload), `${payload.classification} (${(payload.confidence*100).toFixed(1)}%)`);
}

function considerPublish(top, sorted, inferenceMs, frame) {
  const threshold=Number(value("threshold")), required=Number(value("stableFrames")), cooldown=Number(value("cooldown"));
  updateStability(top.className);
  const now=Date.now(), stable=candidateFrames>=required, confident=top.probability>=threshold;
  const changed=top.className!==lastPublishedClass, cooldownPassed=now-lastPublishedAt>=cooldown;
  if (stable && confident && (changed || cooldownPassed)) {
    // Without MQTT, keep classifying but do not try to publish (and do not flood the log).
    if (!mqttClient?.connected) return;
    const payload=buildPayload(top,sorted,inferenceMs,"camera",frame);
    if (publishPayload(payload)) { lastPublishedClass=top.className; lastPublishedAt=now; candidateFrames=0; }
  }
}

async function inferenceLoop() {
  if (!running) return;
  try {
    const start=performance.now();
    const frame=await analyseFrame($("camera"));
    const elapsed=performance.now()-start;
    setText("inferenceTime", `${Math.round(elapsed)} ms`);
    setText("detectedStatus", describeDetection(frame));
    if (frame.predictions) {
      showInputStats(frame.stats);
      const {top,sorted}=showPredictions(frame.predictions);
      drawDetections(frame.detections, frame.detection, top);
      considerPublish(top,sorted,elapsed,frame);
    } else if (frame.mode !== "waiting") {
      showNoItem(); drawDetections(frame.detections);
    }
    inferenceErrors = 0;
  } catch (error) {
    // Tolerate occasional errors (e.g. a frame lost during rotation); stop only if they keep happening.
    inferenceErrors += 1;
    log(`Inference error (${inferenceErrors}/5): ${error.message}`, "error");
    if (inferenceErrors >= 5) { log("Stopping after repeated inference errors.", "error"); stopAll(); return; }
  }
  // Limit load on an old tablet. Approximately 4 inferences/second maximum.
  await new Promise(resolve=>setTimeout(resolve,250));
  animationId=requestAnimationFrame(inferenceLoop);
}

async function start() {
  if (running) return;
  $("startButton").disabled=true; setOverall(model && (detector || !usesDetection()) ? "Starting…" : "Loading models…", "warn");
  try {
    await loadPipeline();
    await startCamera(); running=true; setOverall("Classifying", "ok");
    if (!mqttClient?.connected) log("Classifying without MQTT: results will not be published until MQTT connects.", "warn");
    inferenceLoop();
  } catch (error) { log(error.message,"error"); setOverall("Start failed","bad"); $("startButton").disabled=false; }
}

// ---------- MQTT ----------
function normalizeBrokerUrl(raw) {
  let url = raw.trim();
  if (!url) throw new Error("Broker URL is empty.");
  const scheme = url.match(/^([a-z][a-z0-9+.-]*):\/\//i)?.[1].toLowerCase();
  if (!scheme) {
    url = "wss://" + url.replace(/^\/+/, "");
    log(`No scheme in broker URL; using ${url}`);
  } else if (scheme === "ws" && location.protocol === "http:") {
    log("Using insecure ws:// because this page is not served over HTTPS.", "warn");
  } else if (scheme !== "wss") {
    url = "wss://" + url.slice(scheme.length + 3);
    log(`Browsers can only use WebSockets; changed ${scheme}:// to wss:// (${url}).`, "warn");
  }
  return url;
}

function setMqttButtons(connected) {
  $("testButton").disabled = !connected;
  $("subscribeButton").disabled = !connected;
}

function updateSubscribeUi() {
  $("subscribeButton").textContent = subscribedTopic ? "Unsubscribe" : "Subscribe";
  setText("subscribedStatus", subscribedTopic || "—");
}

function setMqttStatus(status, level="info") {
  if ($("mqttStatus").textContent === status) return;
  setText("mqttStatus", status);
  log(`MQTT ${status.toLowerCase()}.`, level);
}

function disconnectMqtt() {
  if (!mqttClient) return;
  mqttClient.removeAllListeners();
  mqttClient.end(true);
  mqttClient = null; subscribedTopic = "";
  setText("mqttStatus", "Disconnected"); $("mqttButton").textContent = "Connect MQTT";
  setMqttButtons(false); updateSubscribeUi();
  log("MQTT disconnected by user.");
}

function connectMqtt() {
  if (mqttClient) { disconnectMqtt(); return; }
  if (typeof mqtt === "undefined") { log("MQTT library failed to load. Check the internet connection and reload.", "error"); return; }
  let url;
  try { url = normalizeBrokerUrl($("brokerUrl").value); } catch (error) { log(error.message, "error"); return; }
  $("brokerUrl").value = url;
  const options={clientId:`${value("deviceId") || "device"}-web-${Math.random().toString(16).slice(2,10)}`,clean:true,
                 connectTimeout:10000,reconnectPeriod:3000,keepalive:30};
  if (value("mqttUsername")) options.username=value("mqttUsername");
  if ($("mqttPassword").value) options.password=$("mqttPassword").value;
  setText("mqttStatus","Connecting…"); log(`MQTT connecting to ${url}…`);
  $("mqttButton").textContent = "Disconnect MQTT";
  try { mqttClient=mqtt.connect(url,options); }
  catch (error) { log(`MQTT connect failed: ${error.message}`, "error"); mqttClient=null; $("mqttButton").textContent="Connect MQTT"; return; }
  mqttClient.on("connect",()=>{setMqttStatus("Connected"); setMqttButtons(true);});
  mqttClient.on("reconnect",()=>setMqttStatus("Reconnecting…", "warn"));
  mqttClient.on("offline",()=>{setMqttStatus("Offline", "warn"); setMqttButtons(false);});
  mqttClient.on("close",()=>{setMqttStatus("Disconnected", "warn"); setMqttButtons(false);});
  mqttClient.on("error",error=>log(`MQTT error: ${error.message}`,"error"));
  mqttClient.on("message",showReceived);
}

function toggleSubscribe() {
  if (!mqttClient?.connected) { log("Cannot subscribe: MQTT is disconnected.", "error"); return; }
  if (subscribedTopic) {
    const topic = subscribedTopic;
    mqttClient.unsubscribe(topic, error => {
      if (error) return log(`Unsubscribe failed: ${error.message}`, "error");
      subscribedTopic = ""; updateSubscribeUi(); log(`Unsubscribed from ${topic}.`);
    });
    return;
  }
  const topic = value("subscribeTopic");
  if (!topic) { log("Subscribe topic is empty.", "error"); return; }
  mqttClient.subscribe(topic, {qos:1}, (error, granted) => {
    if (error) return log(`Subscribe failed: ${error.message}`, "error");
    if (granted?.[0]?.qos === 128) return log(`Broker rejected subscription to ${topic}.`, "error");
    subscribedTopic = topic; updateSubscribeUi(); log(`Subscribed to ${topic}.`);
  });
}

function prettyPrint(text) {
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

function showReceived(topic, message) {
  received += 1; setText("receivedCount", received);
  const item = document.createElement("li");
  const head = document.createElement("div"), body = document.createElement("pre");
  head.className = "received-head";
  head.textContent = `${new Date().toLocaleTimeString()} · ${topic}`;
  body.textContent = prettyPrint(message.toString());
  item.append(head, body);
  $("receivedLog").prepend(item);
  while ($("receivedLog").children.length > 30) $("receivedLog").lastChild.remove();
}

function clearReceived() {
  $("receivedLog").replaceChildren(); received = 0; setText("receivedCount", 0);
}

function publishTest() {
  const text = $("testMessage").value.trim();
  if (!text) {
    const top={className:"TEST_ONLY",probability:1};
    publishPayload(buildPayload(top,[top],0,"manual_test"));
    return;
  }
  publishText(text, "custom test message");
}

// ---------- Wiring ----------
$("startButton").addEventListener("click",start);
$("loadModelButton").addEventListener("click",loadModelButton);
$("modelSource").addEventListener("change",()=>{updateModelSourceUi(); modelSettingsChanged();});
$("modelFiles").addEventListener("change",()=>{showSelectedFiles(); modelSettingsChanged();});
$("modelUrl").addEventListener("change",modelSettingsChanged);
$("backend").addEventListener("change",backendChanged);
$("pipeline").addEventListener("change",pipelineChanged);
$("fallback").addEventListener("change",updateRegionGuide);
$("region").addEventListener("change",updateRegionGuide);
$("camera").addEventListener("loadedmetadata",updateRegionGuide);
$("camera").addEventListener("resize",updateRegionGuide);   // fires when the phone rotates
window.addEventListener("resize",updateRegionGuide);
$("selfTestButton").addEventListener("click",runSelfTest);
$("photoInput").addEventListener("change",classifyPhoto);
$("normalization").addEventListener("change",modelSettingsChanged);
$("cameraTestButton").addEventListener("click",testCamera);
$("stopButton").addEventListener("click",stopAll);
$("mqttButton").addEventListener("click",connectMqtt);
$("saveButton").addEventListener("click",saveSettings);
$("testButton").addEventListener("click",publishTest);
$("subscribeButton").addEventListener("click",toggleSubscribe);
$("clearReceivedButton").addEventListener("click",clearReceived);
$("logToggle").addEventListener("click",()=>setLogVisible($("logPanel").hidden));
$("clearLogButton").addEventListener("click",clearLog);
$("errorsOnly").addEventListener("change",e=>$("eventLog").classList.toggle("errors-only", e.target.checked));
window.addEventListener("error",e=>log(`Script error: ${e.message}`,"error"));
window.addEventListener("unhandledrejection",e=>log(`Unhandled error: ${e.reason?.message || e.reason}`,"error"));
window.addEventListener("pagehide",()=>{stopAll(); if(mqttClient)mqttClient.end(true);});
loadSettings();
updateModelSourceUi();
pipelineChanged();
log("App ready. Test the camera and MQTT independently, or configure a model URL and start classifying.");
