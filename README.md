# real-time-human-pose-analysis

---

## Spec
CPU: `Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz`
Memory: `16GB RAM`
GPU: `Nvidia GeForce GTX 1660 Ti`

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

---

## Project Structure
- [ ] Version Manager: `Github` (defacto)
- [ ] Library Manager: `uv` (fast)
- Main Functionality
  - [ ] Landmark Detection: `MediaPose`, `YOLO`, `Google MoveNet`, `RTMPose`

---

## Running Code

### Dependencies
- Adding modules: `uv add [module_name]`
- Exporting dependencies: `uv export --format requirements-txt > requirements.txt`
- Importing dependencies: Ensure `uv` exists in device, ensure `pyproject.toml` and `uv.lock` exists in dir, exec `uv sync`
- Dev dependencies: `uv add --dev [module_name]`