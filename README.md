# Week 4 Android tablet waste classifier

This is a no-build browser application for demonstrating on-device image classification and MQTT publication on an old Android tablet. Classification runs in the browser with TensorFlow.js and a Teachable Machine image model. The sample publishes labels, confidence and timing metadata; it does not publish camera images.

## What students learn

1. Load an exported pre-trained model (`model.json`, weights and `metadata.json`).
2. Request the rear camera and run inference locally.
3. Reject low-confidence and unstable predictions.
4. Publish a structured JSON result over MQTT WebSockets.
5. Observe reconnect state, message timing and old-device performance.

## Default model (bundled)

The app loads a lightweight waste classifier from `models/waste-mobilenetv2/` by default, so it works without training anything first.

| Property | Value |
|---|---|
| Architecture | MobileNetV2, width 0.5, 160 × 160 input, ImageNet pretrained then fine-tuned |
| Size | 714k parameters, 1.4 MB download (float16 weights) |
| Classes | `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash` |
| Training data | [TrashNet](https://huggingface.co/datasets/garythung/trashnet) (MIT licence), 2,527 images, 70/15/15 split |
| Held-out test accuracy | 86% overall; cardboard 98%, paper 88%, glass 87%, metal 84%, plastic 83%, trash 57% |

The model needs about 50 million multiply-adds per frame, roughly the same load as the Teachable Machine models. That keeps it usable on older Android tablets.

Limitations to discuss with students:

- TrashNet photos show one item on a white background. Accuracy drops with cluttered scenes, hands in frame or poor lighting.
- There is no `unknown`/`empty` class, so an empty scene is still forced into one of the six classes. The confidence threshold and stable-frame rules reduce, but do not remove, false publications.
- `trash` has few training images and is the weakest class.
- The classes describe materials, not local bins. Map them to your council's bin categories in the MQTT consumer, or train a Teachable Machine model on your own items and bins.

To retrain or change the architecture, run `tools/train_waste_model.py` (TensorFlow 2.15 with Keras 2, plus `tensorflowjs`). The script documents its settings in its header.

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
- Internet access is needed for the CDN libraries, the Teachable Machine model (if used) and the public broker. The bundled default model is served with the app. For offline deployment, download and serve dependencies and model files locally.
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

