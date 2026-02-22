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
  - Body Orientation Estimation:

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
- `multi_headpose.py`
  - Runs any of the `User_??`'s Face video data with mediapose. Extensive testing of mediapose model.

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