# Real-Time Human Pose & Action Analysis 🏃‍♂️🤖

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A high-performance pipeline for real-time human movement analysis, ranging from head-pose estimation to complex body mechanics and temporal action classification.

---

## 💻 System Specification
| Component | Details |
| :--- | :--- |
| **CPU** | Intel(R) Core(TM) i7-10750H @ 2.60GHz |
| **GPU** | NVIDIA GeForce GTX 1660 Ti (6GB VRAM) |
| **RAM** | 16GB |
| **OS** | Windows 11 |

---

## 🎯 Project Roadmap
The goal of this project is to move beyond static landmark detection and achieve **semantic understanding** of human movement.

### Phase 1: Spatial Precision (Completed)
* [x] **Head Pose Estimation:** 3D Landmarks ➔ Euler Angles (Pitch/Yaw/Roll).
* [x] **Body Pose Estimation:** Hybrid Normal-PnP method to bypass the "Planar Trap."
* [x] **Temporal Smoothing:** Exponential Moving Average (EMA) to eliminate landmark jitter.

### Phase 2: Temporal Intelligence (Current)
* [ ] **Action Recognition:** Classifying movements over time (e.g., Squatting, Waving).
* [ ] **Sliding Window Inference:** Processing landmark buffers (30–60 frames).
* [ ] **Multi-Person Tracking:** Maintaining ID consistency across frames.

---

## 🔬 Research & SOTA Benchmarks

### Landmark Detection Comparison
| Feature | MediaPipe | YOLO-Pose | MoveNet | RTMPose |
| :--- | :--- | :--- | :--- | :--- |
| **Landmark Count** | 33 | 17 | 17 | 17 - 133 |
| **3D Support** | ✅ Native | ❌ 2D Only | ❌ 2D Only | ✅ Native |
| **Multi-Person** | Limited | ✅ Yes | ❌ No | ✅ Yes |
| **GPU Efficiency** | High | Medium | High | **SOTA** |

### Action Recognition Methodologies
We utilize **Skeleton-Based HAR**, optimized for the GTX 1660 Ti's CUDA cores.

1.  **RNN/LSTM/GRU:** Best for sequential patterns and low-latency inference.
2.  **ST-GCN (Spatial-Temporal Graph ConvNet):** Treats the body as a graph; excellent for complex mechanics.
3.  **3D-CNN:** Too expensive for this spec (pixel-heavy).

---

1. Recognizing single instance
   - Sliding window on detected instance
   - `LTSM`, accounting for _gradient explosion_
   - `Lightweight ST-GCN` or `ActionMamba`
2. Data
   - [NTU RGB d Skeletons](https://github.com/shahroudy/NTURGB-D/tree/master)
3. Recognizing multiple instances
   - ID and window multiple instances, and batched
   - Perform single instance action recognition for each
4. Multi-instance interactions
   - _Unified Interaction Graph_ instead of isolated independent graphs
   - `Cross-attention` to weight significance of one's node to another
5. Landmark occlusion
   - `Temporal Prediction`: Uses past trajectories
   - `Semantic Prediction`: Inferences most statistically likely positions
   - `Visibility Weighting`: Remove landmarks with low visibility


---

## 🎥 Demos

### 1. Head Pose Estimation (UPNA Gaze Dataset)
*Uses PnP solver + EMA (0.2) to resolve Gimbal Lock.*
![Head Pose Estimation Demo](presentation_mat/UPNA_demo.gif)

### 2. Body Pose Hybrid Estimation (CMU Panoptic)
*Hybrid method: Normal vectors for orientation + PnP for translation.*
![Body Pose Hybrid Estimation Demo](presentation_mat/CMU_hybrid.gif)

---

## 🛠 Project Execution

### Dependency Management (`uv`)
This project uses `uv` for lightning-fast environment resolution.
```bash
# Setup environment
uv sync

# Run specific modules
uv run multi_headpose.py
uv run multi_bodypose.py
```