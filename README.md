# Week 4 Android tablet waste sorter: which bin?

This is a no-build browser application for demonstrating on-device object detection and MQTT publication on an old Android tablet. It shows which kerbside bin an item belongs in: 🔴 red (general waste), 🟡 yellow (recycling) or 🟢 green (food & garden organics). Everything runs in the browser with TensorFlow.js. The app publishes the label, bin, confidence and timing metadata; it does not publish camera images.

## What students learn

1. Load an exported pre-trained model (`model.json`, weights and `metadata.json`).
2. Request the rear camera and run inference locally.
3. Reject low-confidence and unstable predictions.
4. Publish a structured JSON result over MQTT WebSockets.
5. Observe reconnect state, message timing and old-device performance.

## How it works: object → bin

By default (*Pipeline*: **Detect the object, bin from its COCO label**):

1. **Detect.** [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) (lite MobileNetV2, Apache-2.0, `models/coco-ssd-lite/`, 9 MB) finds everyday objects in the camera frame and draws a box around each one.
2. **Pick the item.** Objects mapped to *Ignore* (people and hands, animals, vehicles, furniture) are dropped. The largest remaining object is the item being shown.
3. **Look up the bin.** The object's COCO label (for example `bottle`, `cup`, `banana`, `laptop`) decides the bin through the mapping in [`bins.js`](bins.js).
4. **No object, no message.** If nothing is detected, the app shows **No item** and publishes nothing.

| Bin | Default COCO objects |
|---|---|
| 🟡 Yellow: recycling | bottle, book |
| 🟢 Green: food & garden organics | banana, apple, orange, broccoli, carrot, sandwich, hot dog, pizza, donut, cake |
| 🔴 Red: general waste | cup, wine glass, bowl, cutlery, vase, scissors, toothbrush, soft toys, bags, sports gear |
| ⚪ No kerbside bin: e-waste drop-off | laptop, cell phone, keyboard, mouse, remote, microwave, toaster, hair drier, clock |

Cups go to red because takeaway cups are plastic-lined and mugs are ceramic; drinking glasses are also not kerbside-recyclable. Electronics are not allowed in any kerbside bin, so they get their own grey result rather than red.

**Councils differ.** Edit the defaults in `bins.js`, or open *Bin mapping* in the app and change any object's bin. In-app changes are saved on the device and highlighted; *Reset to defaults* restores `bins.js`. Deciding and justifying these rules for your council is a good student exercise.

**Why not a material classifier?** An earlier version cropped the detected object and asked a waste-material classifier (EcoVision) for its material. The classifier can only answer one of its 10 materials, so for anything else it guesses. In testing, COCO correctly found a `cup` (95%), which the classifier then called `paper` (48%, yellow bin: wrong). COCO's own label is more reliable for deciding the bin. The material-classifier pipelines are still available under *Pipeline* for comparison, and the classifier model is only downloaded when one of them is selected.

**Limitation to discuss with students:** COCO-SSD only knows 80 everyday object classes. It has no class for cans, boxes, crumpled paper or batteries, so these give **No item**. Setting *When nothing is detected* to **Classify the centre region instead** sends those frames to the material classifier; the message then records `"pipeline": "fallback"`.

The detector's weights were stored as float16 with `tools/quantize_tfjs_fp16.py`, halving the download from 18 MB. Its boxes and scores match the original float32 model to within 0.003.

## Bundled material classifiers (optional pipelines)

These are used only by the *detect then classify material* and *classify a fixed region* pipelines, or by the centre-region fallback. Choose one under *Material classifier*.

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

- Neither classifier has an `unknown`/`empty` class. With detection on, an empty scene gives *No item*. With detection off or in fallback mode, an empty scene is still forced into a waste class, usually `cardboard`.
- EcoVision's reported 95% comes from Kaggle datasets that overlap. Expect lower accuracy on your own tablet camera, lighting and backgrounds; measuring this is a good tutorial exercise.
- The classes describe materials, not local bins. Map them to your council's bin categories in the MQTT consumer, or train a Teachable Machine model on your own items and bins.
- Check the **Inference** time on your tablet. If EcoVision is too slow, raise the stable-frame count or switch to Lite.

## Troubleshooting: "everything is cardboard or paper"

This applies to the material-classifier pipelines only; the default COCO → bin pipeline does not use a classifier. With detection off or in fallback mode, both bundled classifiers answer `cardboard` or `paper` when they see little except background. This happens with a black, grey or white frame, a plain wall or table, or random noise. If every item gets these labels, the model is not seeing the item. Work through the panel **What the model sees & diagnostics**:

1. **Check the preview thumbnail.** It shows exactly what the model receives. If it is black or blank, the camera frames are not reaching the model. *Brightness* and *Contrast* are shown under it; contrast below about 8 means a blank frame, and the log warns about this.
2. **Fill the dashed box with the item.** Only the area inside the box is classified. In testing, a bottle occupying about a third of the frame on a plain table was classified as `paper` (whole frame or centre square) but as `plastic` at 99.9% with *zoom 2×*.
3. **Run the model self-test.** It runs the detector and classifier on reference photos and compares the results with known-good values. If it fails, the phone's GPU is computing wrong results: choose **WebAssembly** under *Compute backend* and test again.
4. **Classify a photo.** Take a photo with the phone's camera app and select it. If photos work but live video does not, the problem is in the camera stream rather than the model.

The *Backend* row in *System state* shows the compute backend, GPU name and whether the GPU supports 32-bit or only 16-bit floats.

## Use your own Teachable Machine model

The app has three pipelines (*Pipeline* under *Classifier configuration*):

| Pipeline | What decides the bin | Training needed |
|---|---|---|
| **COCO object detection → bin** (default) | COCO's object label, e.g. `bottle` → yellow | None |
| **Image classifier** | Your Teachable Machine model's class, applied to the dashed target box | Yes |
| **COCO detection, then image classifier** | Your model's class, applied to the object COCO found | Yes |

To use Teachable Machine:

1. Train an image model with one class per item type or bin, **plus an `unknown` (or `background`) class** photographed with no item. Use varied lighting and backgrounds.
2. Export it as TensorFlow.js and copy the shareable link, for example `https://teachablemachine.withgoogle.com/models/MODEL_ID/`. The app adds `model.json` and `metadata.json` automatically. Alternatively, download the export and choose *Upload model files*.
3. Choose *Pipeline* → **Image classifier**, keep *Image classifier* → **Teachable Machine model URL**, paste the link and tap **Load model**.
4. Check **Bin mapping**. The model's classes are listed at the top. Class names containing a bin word are mapped automatically: `recycling`/`recycled`/`yellow` → yellow, `landfill`/`rubbish`/`red` → red, `organics`/`food`/`garden`/`green` → green, material names such as `plastic` or `paper` → yellow, and `unknown`/`background`/`nothing` → **No item**. Set any class marked *No bin rule* by hand; the log lists them after loading.

Model URLs must use `https://` (or `http://localhost` for local testing).

## Run it on an Android tablet

The page must be served over HTTPS for camera access. Opening `index.html` directly from Downloads is not reliable.

1. Upload this folder to GitHub Pages, an SCU HTTPS web server or another static HTTPS host.
2. Open the HTTPS URL in an updated Chrome browser on the tablet.
3. Keep the default COCO pipeline or load your Teachable Machine model (see above), and enter a unique device ID.
4. Use a broker that supports secure MQTT WebSockets. The example public endpoint is for classroom testing only.
5. Set the topic, for example `prog6002/2026/team01-tablet01/classification`.
6. Tap **Connect MQTT**, then **Publish test message**. Confirm it in the HiveMQ WebSocket client.
7. Tap **Start camera & model** and grant camera permission.
8. Present an item steadily and confirm that stable, confident classifications are published.
9. For a display tablet, tap **Tablet mode** in the header. Only the title, the video and the bin result stay on screen, and the page goes full screen where the browser allows it. In landscape, the video and result sit side by side. While stopped, a large **Start** button remains. The setting is remembered on the device; tap **Exit tablet mode** to see the settings again.

Buttons show whether they can be used: clickable buttons are solid and raised; unavailable ones are grey and dashed, with the reason written on them (for example *Publish test message · connect MQTT first*).

## JSON example

```json
{
  "schema_version": 2,
  "message_type": "waste_classification",
  "device_id": "team01-tablet01",
  "sequence": 7,
  "timestamp": "2026-09-19T03:20:10.250Z",
  "source": "camera",
  "pipeline": "coco",
  "bin": "yellow",
  "bin_description": "Recycling",
  "classification": "bottle",
  "confidence": 0.8942,
  "detected_object": {"label": "bottle", "confidence": 0.8942, "bbox": [0.1623, 0.1303, 0.7769, 0.5803]},
  "inference_ms": 96,
  "model": "coco-ssd-lite",
  "alternatives": [{"label": "cup", "confidence": 0.61, "bin": "red"}]
}
```

- `bin`: `red`, `yellow`, `green`, `ewaste`, or `null` when the label has no bin rule.
- `pipeline`: `coco` (COCO label decided the bin), `detect` (object detected, material classifier decided), `fallback` (nothing detected; the centre region was classified), or `classify` (detection switched off).
- `classification` is the label that decided the bin: the COCO object in `coco` mode, otherwise the classifier's material.
- `detected_object` is `null` unless an object was detected. `bbox` is `[x, y, width, height]` as fractions (0–1) of the camera frame.
- `alternatives` lists other detected objects (or runner-up classes) with their bins.
- Schema version 2 added `bin`, `bin_description` and `model` (which replaced `model_url`).

## Stability mechanism

The result is published only when:

- the confidence reaches the configured threshold (default 0.6; COCO scores for clearly visible objects are typically 0.6–0.95);
- the same label and bin occur for the configured number of consecutive frames (default 3); and
- the class has changed, or the cooldown has expired (default 5 seconds).

This reduces flicker and unnecessary MQTT traffic. Students should benchmark thresholds and stable-frame counts against accuracy, latency and message volume rather than copying the defaults without evidence.

## Important Android notes

- Camera access normally requires HTTPS and user permission.
- The app requests 640 x 480 and limits inference to approximately four runs per second to reduce load. Detection plus classification is roughly twice the work of classification alone; check the *Inference* time on your tablet.
- The first start on an old tablet can take several seconds while both models are prepared for the GPU; the status shows *Loading models…*.
- Older tablets may terminate the tab when memory is low. Close other tabs and lower camera resolution or inference frequency if needed.
- Keep the screen awake during demonstrations; browser background tabs are throttled.
- Internet access is needed for the CDN libraries, the Teachable Machine model (if used) and the public broker. The bundled models are served with the app. For offline deployment, download and serve dependencies and model files locally.
- Browser-stored MQTT credentials are visible to the device user. Use a restricted teaching account and topic permissions.

## Suggested tutorial tests

| Test | Evidence |
|---|---|
| Manual MQTT test | JSON appears on the exact subscribed topic |
| Known item under good lighting | Correct class, confidence and inference time |
| Empty scene | *No item*, and nothing is published |
| Bottle, banana, cup, phone | Yellow, green, red and *No kerbside bin* respectively |
| Change a rule in *Bin mapping* | The new bin is shown and published; the row is marked as changed |
| Item COCO does not know (can, box) | *No item*; with the centre-region fallback, a `fallback` message instead |
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

