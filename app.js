"use strict";

const $ = (id) => document.getElementById(id);
const fields = ["modelSource","modelUrl","normalization","deviceId","mqttTopic","subscribeTopic","brokerUrl","threshold","stableFrames","cooldown","mqttUsername"];
const STORAGE_KEY = "prog6002-classifier", LOG_VISIBLE_KEY = "prog6002-log-visible";
let model = null, stream = null, running = false, mqttClient = null, subscribedTopic = "";
let candidate = "", candidateFrames = 0, lastPublishedClass = "", lastPublishedAt = 0;
let sequence = 0, published = 0, received = 0, errorCount = 0, animationId = null;

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
  return { predict: (video) => tm.predict(video, false), getTotalClasses: () => tm.getTotalClasses(),
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

// Lightweight waste model shipped with the app (see tools/train_waste_model.py).
const DEFAULT_MODEL_DIR = "models/waste-mobilenetv2/";

async function loadDefaultModel() {
  if (location.protocol === "file:") throw new Error("The bundled model needs the page served over http(s), not opened as a file.");
  const metaResponse = await fetch(DEFAULT_MODEL_DIR + "metadata.json");
  if (!metaResponse.ok) throw new Error(`Bundled model metadata not found (${metaResponse.status}).`);
  const meta = await metaResponse.json();
  const net = await tf.loadLayersModel(DEFAULT_MODEL_DIR + "model.json");
  return wrapTfjsModel(net, {
    labels: meta.labels ?? [], normalization: meta.normalization ?? "-1to1",
    source: `bundled:${meta.modelName ?? "waste-model"}`, name: `Default waste model (${meta.architecture ?? "bundled"})`
  });
}

// Wraps any TF.js image classifier in the app's predict(video) interface.
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

  function preprocess(video) {
    return tf.tidy(() => {
      let img = tf.browser.fromPixels(video);        // [H, W, 3] int32
      const [h, w] = img.shape, size = Math.min(h, w);
      // Centre-crop to a square (matches Teachable Machine), then resize.
      img = img.slice([Math.floor((h - size) / 2), Math.floor((w - size) / 2), 0], [size, size, 3]);
      img = tf.image.resizeBilinear(img, [height, width]).toFloat();
      if (channels === 1) img = img.mean(2, true);
      if (normalization === "-1to1") img = img.div(127.5).sub(1);
      else if (normalization === "0to1") img = img.div(255);
      return img.expandDims(0);
    });
  }

  async function predict(video) {
    const scores = tf.tidy(() => {
      let out = net.predict(preprocess(video));
      if (Array.isArray(out)) out = out[0];
      out = out.squeeze();
      // Apply softmax if the model outputs logits rather than probabilities.
      const sum = out.sum().dataSync()[0], min = out.min().dataSync()[0];
      return (min < 0 || Math.abs(sum - 1) > 0.01) ? tf.softmax(out) : out;
    });
    const values = await scores.data();
    scores.dispose();
    return Array.from(values, (p, i) => ({ className: labels[i] ?? `Class ${i + 1}`, probability: p }));
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
    const loaders = { default: loadDefaultModel, files: loadModelFiles, url: loadTeachableMachineUrl };
    const loaded = await (loaders[source] ?? loadDefaultModel)();
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

async function loadModelButton() {
  $("loadModelButton").disabled = true;
  try { await loadModel(); } catch (error) { log(error.message, "error"); }
  finally { $("loadModelButton").disabled = false; }
}

function updateModelSourceUi() {
  const source = $("modelSource").value;
  $("defaultSourceFields").hidden = source !== "default";
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

function updateStability(label) {
  if (label === candidate) candidateFrames += 1;
  else { candidate = label; candidateFrames = 1; }
}

function buildPayload(top, sorted, inferenceMs, source="camera") {
  const modelUrl = model?.source ?? null;
  return {
    schema_version:1, message_type:"waste_classification", device_id:value("deviceId"),
    sequence:++sequence, timestamp:new Date().toISOString(), source,
    classification:top.className, confidence:Number(top.probability.toFixed(4)),
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

function considerPublish(top, sorted, inferenceMs) {
  const threshold=Number(value("threshold")), required=Number(value("stableFrames")), cooldown=Number(value("cooldown"));
  updateStability(top.className);
  const now=Date.now(), stable=candidateFrames>=required, confident=top.probability>=threshold;
  const changed=top.className!==lastPublishedClass, cooldownPassed=now-lastPublishedAt>=cooldown;
  if (stable && confident && (changed || cooldownPassed)) {
    const payload=buildPayload(top,sorted,inferenceMs);
    if (publishPayload(payload)) { lastPublishedClass=top.className; lastPublishedAt=now; candidateFrames=0; }
  }
}

async function inferenceLoop() {
  if (!running) return;
  try {
    const start=performance.now();
    const predictions=await model.predict($("camera"));
    const elapsed=performance.now()-start;
    setText("inferenceTime", `${Math.round(elapsed)} ms`);
    const {top,sorted}=showPredictions(predictions);
    considerPublish(top,sorted,elapsed);
  } catch (error) { log(`Inference error: ${error.message}`, "error"); stopAll(); return; }
  // Limit load on an old tablet. Approximately 4 inferences/second maximum.
  await new Promise(resolve=>setTimeout(resolve,250));
  animationId=requestAnimationFrame(inferenceLoop);
}

async function start() {
  if (running) return;
  $("startButton").disabled=true; setOverall("Starting…", "warn");
  try {
    if (!model) await loadModel();
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
log("App ready. Test the camera and MQTT independently, or configure a model URL and start classifying.");
