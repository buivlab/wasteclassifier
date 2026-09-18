"use strict";

const $ = (id) => document.getElementById(id);
const fields = ["modelUrl","deviceId","mqttTopic","brokerUrl","threshold","stableFrames","cooldown","mqttUsername"];
let model = null, stream = null, running = false, mqttClient = null;
let candidate = "", candidateFrames = 0, lastPublishedClass = "", lastPublishedAt = 0;
let sequence = 0, published = 0, animationId = null;

function log(message, level="info") {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} · ${message}`;
  if (level === "error") item.style.color = "#b42318";
  $("eventLog").prepend(item);
  while ($("eventLog").children.length > 30) $("eventLog").lastChild.remove();
}

function setText(id, value) { $(id).textContent = value; }
function setOverall(text, kind="warn") { const el=$("overallStatus"); el.textContent=text; el.className=`pill ${kind}`; }
function value(id) { return $(id).value.trim(); }

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("prog6002-classifier") || "{}");
    fields.forEach(id => { if (saved[id] !== undefined) $(id).value = saved[id]; });
  } catch (error) { log(`Saved configuration ignored: ${error.message}`, "error"); }
}

function saveSettings() {
  const data = Object.fromEntries(fields.map(id => [id, $(id).value]));
  localStorage.setItem("prog6002-classifier", JSON.stringify(data));
  log("Configuration saved on this tablet (password is not stored).")
}

function normalizedModelUrl() {
  const raw = value("modelUrl");
  if (!/^https:\/\//i.test(raw)) throw new Error("Model URL must start with https://");
  return raw.endsWith("/") ? raw : raw + "/";
}

async function loadModel() {
  const base = normalizedModelUrl();
  setText("modelStatus", "Loading…");
  model = await tmImage.load(base + "model.json", base + "metadata.json");
  setText("modelStatus", `${model.getTotalClasses()} classes loaded`);
  log(`Model loaded with ${model.getTotalClasses()} classes.`);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable. Open this page over HTTPS in a current browser.");
  stream = await navigator.mediaDevices.getUserMedia({
    audio:false, video:{facingMode:{ideal:"environment"}, width:{ideal:640}, height:{ideal:480}}
  });
  $("camera").srcObject = stream;
  await $("camera").play();
  $("cameraMessage").classList.add("hidden");
  setText("cameraStatus", "Running");
  log("Rear camera started at a browser-selected resolution.");
}

function stopAll() {
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
}

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
  return {
    schema_version:1, message_type:"waste_classification", device_id:value("deviceId"),
    sequence:++sequence, timestamp:new Date().toISOString(), source,
    classification:top.className, confidence:Number(top.probability.toFixed(4)),
    inference_ms:Math.round(inferenceMs), model_url:normalizedModelUrl(),
    alternatives:sorted.slice(1,3).map(p=>({label:p.className, confidence:Number(p.probability.toFixed(4))}))
  };
}

function publishPayload(payload) {
  if (!mqttClient?.connected) { log("Classification ready but MQTT is disconnected.", "error"); return false; }
  const topic=value("mqttTopic");
  mqttClient.publish(topic, JSON.stringify(payload), {qos:1, retain:false}, error => {
    if (error) return log(`Publish failed: ${error.message}`, "error");
    published += 1; setText("publishedCount", published); setText("payloadPreview", JSON.stringify(payload,null,2));
    log(`Published ${payload.classification} (${(payload.confidence*100).toFixed(1)}%) to ${topic}.`);
  });
  return true;
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
  $("startButton").disabled=true; setOverall("Starting…", "warn");
  try {
    if (!model) await loadModel();
    await startCamera(); running=true; $("stopButton").disabled=false; setOverall("Classifying", "ok");
    inferenceLoop();
  } catch (error) { log(error.message,"error"); setOverall("Start failed","bad"); $("startButton").disabled=false; }
}

function connectMqtt() {
  if (mqttClient) { mqttClient.end(true); mqttClient=null; }
  const url=value("brokerUrl");
  if (!/^wss:\/\//i.test(url)) { log("Use a secure wss:// broker URL on an HTTPS page.","error"); return; }
  const options={clientId:`${value("deviceId")}-web-${Math.random().toString(16).slice(2,10)}`,clean:true,
                 connectTimeout:10000,reconnectPeriod:3000,keepalive:30};
  if (value("mqttUsername")) options.username=value("mqttUsername");
  if ($("mqttPassword").value) options.password=$("mqttPassword").value;
  setText("mqttStatus","Connecting…");
  mqttClient=mqtt.connect(url,options);
  mqttClient.on("connect",()=>{setText("mqttStatus","Connected"); log(`MQTT connected: ${url}`);});
  mqttClient.on("reconnect",()=>setText("mqttStatus","Reconnecting…"));
  mqttClient.on("offline",()=>setText("mqttStatus","Offline"));
  mqttClient.on("close",()=>setText("mqttStatus","Disconnected"));
  mqttClient.on("error",error=>log(`MQTT error: ${error.message}`,"error"));
}

function publishTest() {
  const top={className:"TEST_ONLY",probability:1}, sorted=[top];
  publishPayload(buildPayload(top,sorted,0,"manual_test"));
}

$("startButton").addEventListener("click",start);
$("stopButton").addEventListener("click",stopAll);
$("mqttButton").addEventListener("click",connectMqtt);
$("saveButton").addEventListener("click",saveSettings);
$("testButton").addEventListener("click",publishTest);
window.addEventListener("pagehide",()=>{stopAll(); if(mqttClient)mqttClient.end(true);});
loadSettings();
log("App ready. Configure a model URL, connect MQTT, then start the camera.");
