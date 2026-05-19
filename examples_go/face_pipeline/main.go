// Face detection + recognition (YuNet + ArcFace MobileFaceNet) via od-bridge FFI.
//
// Build & run:
//
//	go run . -detector ../../face_detection_yunet_2023mar.onnx \
//	         -recognizer ../../w600k_mbf.onnx \
//	         -image ../../arnold.jpg
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
	"math"
	"os"
	"unsafe"
)

var landmarkColors = [5]color.RGBA{
	{255, 0, 0, 255},     // left eye - red
	{0, 0, 255, 255},     // right eye - blue
	{255, 255, 0, 255},   // nose - yellow
	{255, 0, 255, 255},   // left mouth - magenta
	{0, 255, 255, 255},   // right mouth - cyan
}

func main() {
	detectorPath := flag.String("detector", "face_detection_yunet_2023mar.onnx", "path to YuNet ONNX model")
	recognizerPath := flag.String("recognizer", "w600k_mbf.onnx", "path to ArcFace ONNX model")
	imagePath := flag.String("image", "arnold.jpg", "path to test image")
	outputPath := flag.String("output", "output.jpg", "path to save result image")
	conf := flag.Float64("conf", 0.7, "confidence threshold")
	nms := flag.Float64("nms", 0.3, "NMS IoU threshold")
	flag.Parse()

	img := loadImage(*imagePath)
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgb := imageToRGB(img)
	fmt.Printf("Image: %dx%d (%d bytes)\n", w, h, len(rgb))

	cDet := C.CString(*detectorPath)
	defer C.free(unsafe.Pointer(cDet))
	cRec := C.CString(*recognizerPath)
	defer C.free(unsafe.Pointer(cRec))

	pipeline := C.face_pipeline_create(cDet, cRec)
	if pipeline == nil {
		fmt.Fprintf(os.Stderr, "failed to create face pipeline\n")
		os.Exit(1)
	}
	defer C.face_pipeline_destroy(pipeline)

	alignedSize := C.face_pipeline_aligned_size(pipeline)
	fmt.Printf("Pipeline created (aligned face size: %dx%d)\n", alignedSize, alignedSize)

	var out C.struct_FaceDetectionResults
	rc := C.face_pipeline_process(
		pipeline,
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		C.int32_t(w), C.int32_t(h),
		C.float(*conf), C.float(*nms),
		&out,
	)
	if rc != C.Ok {
		fmt.Fprintf(os.Stderr, "face pipeline failed: %d\n", rc)
		os.Exit(1)
	}
	defer C.face_pipeline_results_free(&out)

	count := int(out.len)
	fmt.Printf("Faces detected: %d\n", count)

	canvas := toRGBA(img)
	bboxColor := color.RGBA{0, 255, 0, 255}

	if count > 0 {
		faces := unsafe.Slice(out.data, count)
		for i, f := range faces {
			var sumSq float64
			for j := 0; j < 512; j++ {
				v := float64(f.embedding[j])
				sumSq += v * v
			}
			norm := math.Sqrt(sumSq)

			fmt.Printf("  [%d] conf=%.1f%% bbox=(%.1f, %.1f, %.1fx%.1f) embedding L2=%.4f\n",
				i, f.confidence*100,
				f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, norm)

			// Draw bbox
			drawRect(canvas,
				int(f.bbox_x), int(f.bbox_y),
				int(f.bbox_x+f.bbox_w), int(f.bbox_y+f.bbox_h),
				bboxColor)

			// Draw landmarks
			for k := 0; k < 5; k++ {
				lx := int(f.landmarks[k*2])
				ly := int(f.landmarks[k*2+1])
				drawCircle(canvas, lx, ly, 3, landmarkColors[k])
			}
		}

		if count >= 2 {
			fmt.Println("\nPairwise cosine similarities:")
			for i := 0; i < count; i++ {
				for j := i + 1; j < count; j++ {
					var dot float64
					for k := 0; k < 512; k++ {
						dot += float64(faces[i].embedding[k]) * float64(faces[j].embedding[k])
					}
					fmt.Printf("  Face #%d vs Face #%d: %.4f\n", i, j, dot)
				}
			}
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
