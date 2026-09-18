"use strict";

const $ = (id) => document.getElementById(id);
const fields = ["modelUrl","deviceId","mqttTopic","subscribeTopic","brokerUrl","threshold","stableFrames","cooldown","mqttUsername"];
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

// ---------- Camera ----------
function normalizedModelUrl() {
  const raw = value("modelUrl");
  if (!/^https:\/\//i.test(raw)) throw new Error("Model URL must start with https://");
  return raw.endsWith("/") ? raw : raw + "/";
}

async function loadModel() {
  const base = normalizedModelUrl();
  setText("modelStatus", "Loading…");
  try {
    model = await tmImage.load(base + "model.json", base + "metadata.json");
  } catch (error) {
    setText("modelStatus", "Load failed");
    throw new Error(`Model load failed: ${error.message}`);
  }
  setText("modelStatus", `${model.getTotalClasses()} classes loaded`);
  log(`Model loaded with ${model.getTotalClasses()} classes.`);
}

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
  let modelUrl = null;
  try { modelUrl = normalizedModelUrl(); } catch {}
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
    const predictions=await model.predict($("camera"), false);
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
log("App ready. Test the camera and MQTT independently, or configure a model URL and start classifying.");
