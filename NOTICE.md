# Third-Party Notices

GazeControl bundles, downloads, or derives artefacts from the following
open-source projects.  Their original licenses apply to the corresponding
components and override the project-level `LICENSE` file for those parts.

---

## L2CS-Net (gaze estimation model)

- **Source:**     https://github.com/Ahmednull/L2CS-Net
- **Paper:**      *L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained
                  Environments* — Abdelrahman & Hossny (2022),
                  https://arxiv.org/abs/2203.03339
- **License:**    MIT License
- **Copyright:**  Copyright (c) 2022 Ahmed Abdelrahman

GazeControl ships **no L2CS weights**; users download the upstream
`L2CSNet_gaze360.pkl` checkpoint via `tools/download_l2cs.py`, which
also exports it to the ONNX format used at runtime.

```
MIT License

Copyright (c) 2022 Ahmed Abdelrahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## MediaPipe Tasks (hand / face landmarker, blaze-face detector)

- **Source:**     https://github.com/google-ai-edge/mediapipe
- **License:**    Apache License 2.0
- **Copyright:**  Copyright (c) The MediaPipe Authors

GazeControl downloads `hand_landmarker.task`, `face_landmarker.task`, and
`blaze_face_short_range.tflite` from the official Google CDN at first
run via `gazecontrol.utils.model_downloader` (SHA-256 pinned).
Full license: https://www.apache.org/licenses/LICENSE-2.0
