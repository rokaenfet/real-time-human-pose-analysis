# real-time-human-pose-analysis

---

## Spec
CPU: `Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz`
Memory: `16GB RAM`
GPU: `Nvidia GeForce GTX 1660 Ti`

## OS
`Windows 11`

---

## Project Guideline
Real-time video based pose analysis of humans

## Aim
Show that one can automatically look-out for specific movements of a human

---

## Research and Notes

### CUDA-Torch Compat
> Python 3.12
> https://download.pytorch.org/whl/cu124

### Landmark Detection
| Feature | MediaPipe | YOLOv8/11/26-Pose | MoveNet | RTMPose |
|---|---|---|---|---|
| Landmark | Count | 33 | 17 | 17 | |17 - 133 |
| Multi-Person? | Limited | ✅ Yes | No | ✅ Yes |
| 3D Support? | ✅ Yes | No (2D) | No (2D) | ✅ Yes |

[Google MediaPipe Doc](https://ai.google.dev/edge/mediapipe/solutions/guide)

### Testing Materials
- Me
- [UPNA](https://www.unavarra.es/gi4e/databases/hpdb?languageId=1) Gaze interaciton Dataset

---

## Project Structure
- [ ] Version Manager: `Github` (defacto)
- [ ] Library Manager: `uv` (fast)
- Main Functionality
  - Landmark Detection:
    - [ ] `MediaPose`
    - [ ] `YOLO`
    - [ ] `Google MoveNet`
    - [ ] `RTMPose`
  - Head Pose Estimation: 
    - [ ] `Vectory Heuristics`
    - [ ] `Centroid of Upper Body (L Shoulder, R Shoulder, Nose)`
    - [ ] `3D Landmarks > Euler`
  - Body Pose Estimation:
    - [ ] `Normal Vector`
      - [ ] No Coordinate position
    - [ ] `Euler`
      - [ ] Flat Body Coordinate Acquirement is too ambiguous, no rigid static body.
    - [ ] `Hybrid`
      - [ ] 


### Mediapipe + Euler + EMA + UPNA
![Head Pose Estimation Demo](presentation_mat/UPNA_demo.gif)

### Mediapipe + Norm Vecotr + CMU
![Body Pose Normal Vector Estimation Demo](presentation_mat/CMU_norm_vector.gif)
---

## Running Code

### Dependencies
- Adding modules: `uv add [module_name]`
- Exporting dependencies: `uv export --format requirements-txt > requirements.txt`
- Importing dependencies: Ensure `uv` exists in device, ensure `pyproject.toml` and `uv.lock` exists in dir, exec `uv sync`
- Using uv dependencies: `uv run [code].py`
- Dev dependencies: `uv add --dev [module_name]`

### MediaPipe Models
Pose Landmark: `curl -o pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`
Face Landmark: `curl -o face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`


### Code
- `./helper/`
  - Helper libs

- `multi_headpose.py`
  - Runs any of the `User_??`'s Face video data with mediapose. Extensive testing of mediapose model.

**FaceLandMarkerResult**
```
FaceLandmarkerResult:
  face_landmarks:
    NormalizedLandmark #0:
      x: 0.5971359014511108
      y: 0.485361784696579
      z: -0.038440968841314316
      visibility: None
      presence: None
      name: None
    NormalizedLandmark #1:
      x: 0.3302789330482483
      y: 0.29289937019348145
      z: -0.09489090740680695
      visibility: None
      presence: None
      name: None
    ... (478 landmarks for each face)
  face_blendshapes:
    browDownLeft: 0.8296722769737244
    browDownRight: 0.8096957206726074
    browInnerUp: 0.00035583582939580083
    browOuterUpLeft: 0.00035752105759456754
    ... (52 blendshapes for each face)
  facial_transformation_matrixes:
    [9.99158978e-01, -1.23036895e-02, 3.91213447e-02, -3.70770246e-01]
    [1.66496094e-02,  9.93480563e-01, -1.12779640e-01, 2.27719707e+01]
```

**PoseLandmarkerResult**
```PoseLandmarkerResult
PoseLandmarkerResult:
  Landmarks:
    Landmark #0:
      x            : 0.638852
      y            : 0.671197
      z            : 0.129959
      visibility   : 0.9999997615814209
      presence     : 0.9999984502792358
    Landmark #1:
      x            : 0.634599
      y            : 0.536441
      z            : -0.06984
      visibility   : 0.999909
      presence     : 0.999958
    ... (33 landmarks per pose)
  WorldLandmarks:
    Landmark #0:
      x            : 0.067485
      y            : 0.031084
      z            : 0.055223
      visibility   : 0.9999997615814209
      presence     : 0.9999984502792358
    Landmark #1:
      x            : 0.063209
      y            : -0.00382
      z            : 0.020920
      visibility   : 0.999976
      presence     : 0.999998
    ... (33 world landmarks per pose)
  SegmentationMasks:
    ... (pictured below)
```
---

## Note.

`mediapipe`
```py
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
class FaceLandmarkerResult:
  """The face landmarks detection result from FaceLandmarker, where each vector element represents a single face detected in the image.

  Attributes:
    face_landmarks: Detected face landmarks in normalized image coordinates.
    face_blendshapes: Optional face blendshapes results.
    facial_transformation_matrixes: Optional facial transformation matrix.
  """
```
- `x,y` coords normalized to frame resolution, `z` is relative to _face center_ which is not canonical with the `x,y`.
- `mediapipe` coord system for + dir, `x` to the right, `y` down, `z` away from camera
- `mediapipe` > face landmark normalized `x y z` > actual `x y` > rigit head model fitting coord system > `PnP` > Pitch Yaw Roll

**HOWEVER** Since `PnP` is an optimzation problem, it may get stuck in a local minimum, resulting in bimodal results.
- Apply some `Temporal Smoothing` > `Exponential Moving Average`, Use different PnP solver `SOLVEPNP_EPNP`
  - ?
- Noise when looking around. 
  - Chosen landmark-of-interest must be rigid parts of the face (bones, eyes, nose, etc...)
- Jump in `pitch` when turning head left to right. `Gimbal Lock`. 
- Default pitch when looking roughly straight at `~54`. Static offset?
- **SOLVED**: Understanding the 3D coordinate space of all inputs and output fixes.



```py
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                 (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)])

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5),
                           (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97),
                           (97, 2), (2, 326), (326, 327), (327, 294),
                           (294, 278), (278, 344), (344, 440), (440, 275),
                           (275, 4), (4, 45), (45, 220), (220, 115), (115, 48),
                           (48, 64), (64, 98)])
```