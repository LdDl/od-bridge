# Go Examples for od-bridge

These examples demonstrate calling od-bridge from Go via CGO.

## Prerequisites

1. Build od-bridge:

```bash
cd /path/to/od-bridge
cargo build --release
```

2. Install the library system-wide:

```bash
sudo mkdir -p /usr/local/include/od-bridge
sudo cp od_bridge.h /usr/local/include/od-bridge/

PC_DIR=$(pkg-config --variable pc_path pkg-config | cut -d: -f1)
sudo cp od_bridge.pc "$PC_DIR/"

sudo cp target/release/libod_bridge.so /usr/local/lib/
sudo ldconfig
```

3. Verify:

```bash
pkg-config --cflags --libs od_bridge
```

## Examples

### YOLO Object Detection

```bash
cd yolo_detect
go run . -model ../../yolov4-tiny.onnx -image ../../dog.jpg -width 416 -height 416
```

### Face Pipeline (YuNet + ArcFace MobileFaceNet)

```bash
cd face_pipeline
go run . -detector ../../face_detection_yunet_2023mar.onnx \
         -recognizer ../../w600k_mbf.onnx \
         -image ../../arnold.jpg
```

### Face Pipeline with ResNet50

```bash
cd face_pipeline_r50
go run . -detector ../../face_detection_yunet_2023mar.onnx \
         -recognizer ../../w600k_r50.onnx \
         -image ../../arnold.jpg
```

### Face Pipeline with YuNet-Nano (multi-face)

```bash
cd face_yunet_nano
go run . -detector ../../yunet_n_320_320.onnx \
         -recognizer ../../w600k_mbf.onnx \
         -image ../../oscar_selfies.jpg
```

## Running without pkg-config

If od-bridge is not installed system-wide, you can point CGO directly at the build directory:

```bash
CGO_LDFLAGS="-L../../target/release -lod_bridge -lm -ldl -lpthread" \
CGO_CFLAGS="-I../../" \
LD_LIBRARY_PATH=../../target/release \
go run . [flags...]
```

## Models

| File | Description |
|------|-------------|
| `yolov4-tiny.onnx` | YOLOv4-tiny object detection (80 COCO classes) |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (standard) |
| `yunet_n_320_320.onnx` | YuNet-Nano face detector (lightweight, 320x320) |
| `w600k_mbf.onnx` | ArcFace MobileFaceNet (fast, 112x112 input) |
| `w600k_r50.onnx` | ArcFace ResNet50 (accurate, 112x112 input) |
