# Week 4 Android tablet waste classifier

This is a no-build browser application for demonstrating on-device image classification and MQTT publication on an old Android tablet. Classification runs in the browser with TensorFlow.js and a Teachable Machine image model. The sample publishes labels, confidence and timing metadata; it does not publish camera images.

## What students learn

1. Load an exported pre-trained model (`model.json`, weights and `metadata.json`).
2. Request the rear camera and run inference locally.
3. Reject low-confidence and unstable predictions.
4. Publish a structured JSON result over MQTT WebSockets.
5. Observe reconnect state, message timing and old-device performance.

## Bundled models

The app ships with two waste classifiers, so it works without training anything first. Choose one under *Model source*.

| | **Default: EcoVision MobileNetV3** | **Lite: TrashNet MobileNetV2** |
|---|---|---|
| Folder | `models/ecovision-mobilenetv3/` | `models/waste-mobilenetv2/` |
| Architecture | MobileNetV3-Large, 224 × 224 input | MobileNetV2 width 0.5, 160 × 160 input |
| Size | 4.2M parameters, 8.4 MB download | 714k parameters, 1.4 MB download |
| Compute per frame | ≈ 220M multiply-adds | ≈ 50M multiply-adds (about 5× less) |
| Classes | `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash` | `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash` |
| Training data | ≈ 20,000 varied photos ([Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)) | 2,527 photos on a white background ([TrashNet](https://huggingface.co/datasets/garythung/trashnet)) |
| Reported accuracy | ≈ 95% (author's test set) | 86% (held-out test set) |
| Source and licence | [AmadFR/ecovision_mobilenetv3](https://huggingface.co/AmadFR/ecovision_mobilenetv3), MIT | Trained for this unit, `tools/train_waste_model.py` |

**EcoVision (the default)** was published as a PyTorch model. `tools/convert_ecovision.py` converts it: PyTorch → ONNX → TensorFlow (onnx2tf) → TF.js graph model with float16 weights. The ImageNet normalisation and softmax are built into the graph. On the author's example images, the converted TF.js model gave the same top class as the original PyTorch model every time, with probabilities within 0.006.

**Lite** is only for tablets too slow to run EcoVision. It was trained on photos of single items on a white background, so in real camera scenes it over-predicts `cardboard` and `paper`.

Limitations to discuss with students:

- Neither model has an `unknown`/`empty` class, so an empty scene is still forced into a waste class. The confidence threshold and stable-frame rules reduce, but do not remove, false publications.
- EcoVision's reported 95% comes from Kaggle datasets that overlap. Expect lower accuracy on your own tablet camera, lighting and backgrounds; measuring this is a good tutorial exercise.
- The classes describe materials, not local bins. Map them to your council's bin categories in the MQTT consumer, or train a Teachable Machine model on your own items and bins.
- Check the **Inference** time on your tablet. If EcoVision is too slow, raise the stable-frame count or switch to Lite.

## Troubleshooting: "everything is cardboard or paper"

Both bundled models answer `cardboard` or `paper` when they see little except background. This happens with a black, grey or white frame, a plain wall or table, or random noise. If every item gets these labels, the model is not seeing the item. Work through the panel **What the model sees & diagnostics**:

1. **Check the preview thumbnail.** It shows exactly what the model receives. If it is black or blank, the camera frames are not reaching the model. *Brightness* and *Contrast* are shown under it; contrast below about 8 means a blank frame, and the log warns about this.
2. **Fill the dashed box with the item.** Only the area inside the box is classified. In testing, a bottle occupying about a third of the frame on a plain table was classified as `paper` (whole frame or centre square) but as `plastic` at 99.9% with *zoom 2×*.
3. **Run the model self-test.** It classifies three reference photos and compares every probability with known-good results. If it fails, the phone's GPU is computing wrong results: choose **WebAssembly** under *Compute backend* and test again.
4. **Classify a photo.** Take a photo with the phone's camera app and select it. If photos work but live video does not, the problem is in the camera stream rather than the model.

The *Backend* row in *System state* shows the compute backend, GPU name and whether the GPU supports 32-bit or only 16-bit floats.

## Prepare your own model

Select **Teachable Machine URL** or **Upload model files** under *Model source* to use your own model instead.

Create or obtain a Teachable Machine image model with classes that match the project, for example `recycled`, `green`, `landfill` and `unknown`. Export it as TensorFlow.js and host it with HTTPS. A shared Teachable Machine URL has this form:

`https://teachablemachine.withgoogle.com/models/MODEL_ID/`

The app adds `model.json` and `metadata.json` automatically. Include an `unknown`/`no item` class and diverse lighting/background examples; otherwise a classifier is forced to choose a waste class for every frame.

## Run it on an Android tablet

The page must be served over HTTPS for camera access. Opening `index.html` directly from Downloads is not reliable.

1. Upload this folder to GitHub Pages, an SCU HTTPS web server or another static HTTPS host.
2. Open the HTTPS URL in an updated Chrome browser on the tablet.
3. Keep the default model, or choose another model source, and enter a unique device ID.
4. Use a broker that supports secure MQTT WebSockets. The example public endpoint is for classroom testing only.
5. Set the topic, for example `prog6002/2026/team01-tablet01/classification`.
6. Tap **Connect MQTT**, then **Publish test message**. Confirm it in the HiveMQ WebSocket client.
7. Tap **Start camera & model** and grant camera permission.
8. Present an item steadily and confirm that stable, confident classifications are published.

## JSON example

```json
{
  "schema_version": 1,
  "message_type": "waste_classification",
  "device_id": "team01-tablet01",
  "sequence": 7,
  "timestamp": "2026-09-18T03:20:10.250Z",
  "source": "camera",
  "classification": "recycled",
  "confidence": 0.9341,
  "inference_ms": 186,
  "model_url": "https://teachablemachine.withgoogle.com/models/MODEL_ID/",
  "alternatives": [{"label":"landfill","confidence":0.0412}]
}
```

## Stability mechanism

The result is published only when:

- the top confidence reaches the configured threshold (default 0.75);
- the same top class occurs for the configured number of consecutive frames (default 3); and
- the class has changed, or the cooldown has expired (default 5 seconds).

This reduces flicker and unnecessary MQTT traffic. Students should benchmark thresholds and stable-frame counts against accuracy, latency and message volume rather than copying the defaults without evidence.

## Important Android notes

- Camera access normally requires HTTPS and user permission.
- The app requests 640 x 480 and limits inference to approximately four runs per second to reduce load.
- Older tablets may terminate the tab when memory is low. Close other tabs and lower camera resolution or inference frequency if needed.
- Keep the screen awake during demonstrations; browser background tabs are throttled.
- Internet access is needed for the CDN libraries, the Teachable Machine model (if used) and the public broker. The bundled models are served with the app. For offline deployment, download and serve dependencies and model files locally.
- Browser-stored MQTT credentials are visible to the device user. Use a restricted teaching account and topic permissions.

## Suggested tutorial tests

| Test | Evidence |
|---|---|
| Manual MQTT test | JSON appears on the exact subscribed topic |
| Known item under good lighting | Correct class, confidence and inference time |
| Empty scene / unknown object | `unknown` class rather than a forced waste class |
| Confidence just below threshold | No publication |
| Rapid class changes | Consecutive-frame rule suppresses flicker |
| Network disconnection | MQTT state changes and reconnects after restoration |
| Old tablet benchmark | Median inference time and approximate messages/minute |

## Production limitations

This is a teaching prototype. A production system needs a private authenticated broker, TLS certificate validation, restrictive topic access, locally hosted/pinned dependencies, model version identifiers, privacy review, device management and stronger offline handling. Do not publish images, personal information or precise location data without a documented need and appropriate consent.

## Upstream documentation

- Teachable Machine community image library: https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image
- TensorFlow.js: https://www.tensorflow.org/js
- MQTT.js browser client: https://github.com/mqttjs/MQTT.js
- HiveMQ WebSocket client: https://www.hivemq.com/demos/websocket-client/

