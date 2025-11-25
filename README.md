# Traffic YOLO Detect

## Objective
Deploy a fine-tuned YOLO-based pipeline that detects and tracks four vehicle classes (Bus, Car, Motor, Truck) in traffic imagery and video, enabling downstream analytics such as flow counting, congestion alerts, or safety monitoring. The workflow spans dataset validation, training, evaluation, and real-time inference so teams can reproduce experiments and ship performant checkpoints quickly.

## Dataset
- Source: [Vietnamese Vehicle Dataset on Roboflow](https://universe.roboflow.com/car-classification/vietnamese-vehicle)
- Classes: Bus, Car, Motor, Truck (`nc = 4`)
- Splits: train/val/test image folders with paired label directories inferred automatically
- Preprocessing utilities: integrity checks for image-label pairs, class distribution summaries, and bounding-box clipping/remapping helpers

## Methodology
- Data validation via `data/data_preprocessor.py`:
  - `load_data_set`, `verify_image_label_pairs`, and `compute_dataset_stats` guard against missing labels and surface imbalance/small-box ratios
  - `clip_and_fix_boxes` clamps boxes to `[0,1]` and drops degenerate annotations
  - `remap_yolo_labels` updates legacy IDs to the four target classes
- Training orchestrated by `model/model_trainer.py`:
  - Loads base weights (e.g., Ultralytics YOLOv8) and hyperparameters from `configs/training_config.yaml`
  - Runs `model.train(**config)`, persists experiment artifacts under `runs/detect/train/`, and returns the `best.pt` checkpoint
  - Provides `valid_model` for evaluation and `benchmark_on_video` to measure FPS on real footage
- Inference handled by `model/yolo_detector.py`:
  - Wraps `YOLO.track(..., persist=True)` to emit per-frame detections with IDs, scores, and class names
  - Supports batch frame processing, video sampling with stride/max frame limits, and optional bounding-box rendering

## Model Evaluation
- Training logs: metrics, confusion matrices, precision/recall curves, batch previews located in `runs/detect/train/`
- Key artifacts include `results.csv`, `results.png`, and normalized confusion matrices for diagnosing per-class performance
- Validation via `model.valid_model()` uses Ultralytics’ standard reporting (mAP50, mAP50-95, precision, recall)
- `benchmark_on_video` prints aggregate FPS to gauge deployment readiness on specific hardware

## Deliverables
- `configs/model_config.yaml`: runtime settings (weights path, device, confidence threshold, class names, optional tracking flag)
- `configs/training_config.yaml`: training hyperparameters, augmentation knobs, and dataset reference
- `data/data_preprocessor.py`: dataset QA, statistics, label fixing, and visualization helpers
- `model/model_trainer.py`: training/validation entry points and video benchmarking
- `model/yolo_detector.py`: inference/tracking wrapper with optional visualization utilities
- `runs/detect/train/weights/best.pt`: latest fine-tuned checkpoint ready for deployment
- `training.ipynb`: interactive notebook for experimentation (data inspection, training, or inference demos)

## Business Insights
- Reliable per-class tracking in dense traffic enables accurate vehicle counts and dwell-time metrics, supporting congestion analytics
- Small-box ratios and class distribution stats highlight data gaps, steering additional data collection for underrepresented classes
- Real-time FPS benchmarking surfaces the hardware needed for roadside deployments versus offline analytics pipelines
- Consistent experiment artifacts make it easy to audit improvements across hyperparameter sweeps or dataset refreshes

## Tech Stack
Python, Ultralytics YOLO, PyYAML, OpenCV, NumPy, Matplotlib, Torch (via Ultralytics), and supporting libraries for dataset analysis.

