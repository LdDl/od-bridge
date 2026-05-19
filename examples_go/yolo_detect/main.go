// YOLO object detection via od-bridge FFI.
//
// Build & run:
//
//	# Option A: pkg-config (if od-bridge is installed system-wide)
//	go run . -model ../../yolov4-tiny.onnx -image ../../dog.jpg -width 416 -height 416
//
//	# Option B: manual paths
//	CGO_LDFLAGS="-L../../target/release -lod_bridge -lm -ldl -lpthread" \
//	CGO_CFLAGS="-I../../" \
//	LD_LIBRARY_PATH=../../target/release \
//	go run . -model ../../yolov4-tiny.onnx -image ../../dog.jpg -width 416 -height 416
package main

/*
#cgo pkg-config: od_bridge
#include "od_bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"os"
	"unsafe"
)

func main() {
	modelPath := flag.String("model", "model.onnx", "path to ONNX model")
	imagePath := flag.String("image", "dog.jpg", "path to test image")
	outputPath := flag.String("output", "output.jpg", "path to save result image")
	inputW := flag.Uint("width", 416, "model input width")
	inputH := flag.Uint("height", 416, "model input height")
	conf := flag.Float64("conf", 0.3, "confidence threshold")
	nms := flag.Float64("nms", 0.4, "NMS IoU threshold")
	flag.Parse()

	// Load image
	img := loadImage(*imagePath)
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgb := imageToRGB(img)
	fmt.Printf("Image: %dx%d (%d bytes)\n", w, h, len(rgb))

	// Create model
	cModel := C.CString(*modelPath)
	defer C.free(unsafe.Pointer(cModel))

	handle := C.od_model_create(cModel, C.uint32_t(*inputW), C.uint32_t(*inputH))
	if handle == nil {
		fmt.Fprintf(os.Stderr, "failed to load model: %s\n", *modelPath)
		os.Exit(1)
	}
	defer C.od_model_free(handle)
	fmt.Printf("Model loaded: %s\n", *modelPath)

	// Run detection
	var out C.struct_OdDetections
	rc := C.od_model_detect(
		handle,
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int32_t(w), C.int32_t(h),
		C.float(*conf), C.float(*nms),
		&out,
	)
	if rc != C.Ok {
		fmt.Fprintf(os.Stderr, "detection failed: %d\n", rc)
		os.Exit(1)
	}
	defer C.od_detections_free(&out)

	count := int(out.len)
	fmt.Printf("Detections: %d\n", count)

	// Draw results
	canvas := toRGBA(img)
	bboxColor := color.RGBA{0, 255, 0, 255}

	if count > 0 {
		results := unsafe.Slice(out.data, count)
		for i, d := range results {
			fmt.Printf("  [%d] class=%d conf=%.1f%% bbox=(%d, %d, %dx%d)\n",
				i, d.class_id, d.confidence*100,
				d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h)

			drawRect(canvas,
				int(d.bbox_x), int(d.bbox_y),
				int(d.bbox_x)+int(d.bbox_w), int(d.bbox_y)+int(d.bbox_h),
				bboxColor)
		}
	}

	saveJPEG(canvas, *outputPath)
	fmt.Printf("Saved to %s\n", *outputPath)
	fmt.Println("Done.")
}

func loadImage(path string) image.Image {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open %s: %v\n", path, err)
		os.Exit(1)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to decode %s: %v\n", path, err)
		os.Exit(1)
	}
	return img
}

func imageToRGB(img image.Image) []byte {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgb := make([]byte, w*h*3)

	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rgb[idx] = byte(r >> 8)
			rgb[idx+1] = byte(g >> 8)
			rgb[idx+2] = byte(b >> 8)
			idx += 3
		}
	}
	return rgb
}

func toRGBA(img image.Image) *image.RGBA {
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
	return rgba
}

func drawRect(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	b := img.Bounds()
	x0 = clamp(x0, b.Min.X, b.Max.X-1)
	y0 = clamp(y0, b.Min.Y, b.Max.Y-1)
	x1 = clamp(x1, b.Min.X, b.Max.X-1)
	y1 = clamp(y1, b.Min.Y, b.Max.Y-1)

	for x := x0; x <= x1; x++ {
		img.SetRGBA(x, y0, c)
		img.SetRGBA(x, y1, c)
	}
	for y := y0; y <= y1; y++ {
		img.SetRGBA(x0, y, c)
		img.SetRGBA(x1, y, c)
	}
}

func drawCircle(img *image.RGBA, cx, cy, r int, c color.RGBA) {
	b := img.Bounds()
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			if dx*dx+dy*dy <= r*r {
				px, py := cx+dx, cy+dy
				if px >= b.Min.X && px < b.Max.X && py >= b.Min.Y && py < b.Max.Y {
					img.SetRGBA(px, py, c)
				}
			}
		}
	}
}

func saveJPEG(img image.Image, path string) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create %s: %v\n", path, err)
		os.Exit(1)
	}
	defer f.Close()
	if err := jpeg.Encode(f, img, &jpeg.Options{Quality: 95}); err != nil {
		fmt.Fprintf(os.Stderr, "failed to encode %s: %v\n", path, err)
		os.Exit(1)
	}
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
