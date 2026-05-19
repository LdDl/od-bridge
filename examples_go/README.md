# Go Examples for od-bridge

These examples demonstrate calling od-bridge from Go via CGO.

## Prerequisites

### Option A: Install from source

1. Build od-bridge (requires Rust 1.85+):

```bash
cd /path/to/od-bridge
cargo build --release
```

2. Install the library system-wide:

```bash
sudo mkdir -p /usr/local/include/od-bridge
sudo cp od_bridge.h /usr/local/include/od-bridge/

sudo cp target/release/libod_bridge.so /usr/local/lib/
sudo cp target/release/libod_bridge.a  /usr/local/lib/

# Generate and install pkg-config file
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
sed -e "s|@PREFIX@|/usr/local|" -e "s|@VERSION@|${VERSION}|" \
  od_bridge.pc.in | sudo tee /usr/local/lib/pkgconfig/od_bridge.pc > /dev/null

# Ensure /usr/local/lib is in the linker search path (needed on some distros, e.g. Arch)
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf
sudo ldconfig
```

### Option B: Install from pre-built release (no Rust required)

```bash
curl -L https://github.com/LdDl/od-bridge/releases/download/vX.Y.Z/od-bridge-linux-amd64.tar.gz | tar xz
cd od-bridge-linux-amd64
./install.sh
```

### Verify

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

All examples save visual results to `output.jpg` (use `-output` flag to change the path).

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
