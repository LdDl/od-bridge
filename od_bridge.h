#ifndef OD_BRIDGE_H
#define OD_BRIDGE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Error code returned by all functions.
 */
typedef enum OdError {
  /**
   * No error.
   */
  Ok = 0,
  /**
   * Null pointer or invalid dimension passed.
   */
  InvalidArgument = 1,
  /**
   * ONNX model file could not be loaded.
   */
  ModelLoadFailed = 2,
  /**
   * Inference failed at runtime.
   */
  DetectionFailed = 3,
  /**
   * RGB pixel buffer could not be converted to Array3.
   */
  ImageConvertFailed = 4,
} OdError;

/**
 * Opaque face pipeline handle. Created by `face_pipeline_create*`.
 */
typedef struct FacePipelineHandle FacePipelineHandle;

/**
 * Opaque model handle. Created by any `od_model_create_*` function.
 */
typedef struct ModelHandle ModelHandle;

/**
 * Single detection result (flat, no pointers, safe for CGO memcpy).
 */
typedef struct OdDetection {
  /**
   * Top-left corner X coordinate (pixels).
   */
  int32_t bbox_x;
  /**
   * Top-left corner Y coordinate (pixels).
   */
  int32_t bbox_y;
  /**
   * Bounding box width (pixels).
   */
  int32_t bbox_w;
  /**
   * Bounding box height (pixels).
   */
  int32_t bbox_h;
  /**
   * Predicted class index (zero-based).
   */
  int32_t class_id;
  /**
   * Detection confidence in [0.0, 1.0].
   */
  float confidence;
} OdDetection;

/**
 * Detection results batch. Caller must free via `od_detections_free`.
 */
typedef struct OdDetections {
  /**
   * Pointer to the first element of the results array.
   */
  struct OdDetection *data;
  /**
   * Number of detections in the array.
   */
  int32_t len;
} OdDetections;

/**
 * Single face detection + recognition result (flat, safe for CGO memcpy).
 */
typedef struct FaceDetectionResult {
  /**
   * Bounding box top-left X (pixels).
   */
  float bbox_x;
  /**
   * Bounding box top-left Y (pixels).
   */
  float bbox_y;
  /**
   * Bounding box width (pixels).
   */
  float bbox_w;
  /**
   * Bounding box height (pixels).
   */
  float bbox_h;
  /**
   * Detection confidence in [0.0, 1.0].
   */
  float confidence;
  /**
   * 5 facial landmarks: [x0,y0, x1,y1, ..., x4,y4] (10 floats).
   */
  float landmarks[10];
  /**
   * 512-dimensional L2-normalized embedding.
   */
  float embedding[512];
} FaceDetectionResult;

/**
 * Face detection results batch. Caller must free via `face_pipeline_results_free`.
 */
typedef struct FaceDetectionResults {
  /**
   * Pointer to the first element of the results array.
   */
  struct FaceDetectionResult *data;
  /**
   * Number of face detections in the array.
   */
  int32_t len;
} FaceDetectionResults;

/**
 * Create a model from an ONNX file (ORT backend, CPU).
 *
 * # Parameters
 * - `model_path`: null-terminated path to `.onnx` file
 * - `input_w`, `input_h`: model input dimensions (e.g. 416, 416)
 *
 * # Returns
 * Opaque pointer, or null on error.
 *
 * # Safety
 * `model_path` must be a valid null-terminated C string.
 */
struct ModelHandle *od_model_create(const char *model_path, uint32_t input_w, uint32_t input_h);

/**
 * Create a model from an ONNX file with CUDA execution provider.
 *
 * # Safety
 * `model_path` must be a valid null-terminated C string.
 */
struct ModelHandle *od_model_create_cuda(const char *model_path,
                                         uint32_t input_w,
                                         uint32_t input_h);

/**
 * Create a model from an ONNX file with TensorRT execution provider (via ORT).
 *
 * # Safety
 * `model_path` must be a valid null-terminated C string.
 */
struct ModelHandle *od_model_create_tensorrt(const char *model_path,
                                             uint32_t input_w,
                                             uint32_t input_h);

/**
 * Create a model from a serialized TensorRT engine file (native TensorRT, no ORT).
 *
 * # Safety
 * `engine_path` must be a valid null-terminated C string.
 */
struct ModelHandle *od_model_create_trt(const char *engine_path);

/**
 * Create a model from an RKNN model file (Rockchip NPU).
 *
 * # Parameters
 * - `model_path`: null-terminated path to `.rknn` file
 * - `num_classes`: number of classes the model was trained on
 *
 * # Safety
 * `model_path` must be a valid null-terminated C string.
 */
struct ModelHandle *od_model_create_rknn(const char *model_path, uint32_t num_classes);

/**
 * Free a model handle.
 *
 * # Safety
 * `handle` must have been returned by `od_model_create*` and not yet freed.
 */
void od_model_free(struct ModelHandle *handle);

/**
 * Run detection on an RGB image.
 *
 * Works with any backend: the handle dispatches to the correct runtime internally.
 *
 * # Parameters
 * - `handle`: model handle from any `od_model_create_*` function
 * - `pixels_rgb`: pointer to `width * height * 3` bytes (RGB, row-major, HWC)
 * - `img_w`, `img_h`: image dimensions in pixels
 * - `conf_threshold`: confidence threshold (e.g. 0.3)
 * - `nms_threshold`: NMS IoU threshold (e.g. 0.4)
 * - `out`: pointer to `OdDetections` struct, filled on success
 *
 * # Returns
 * `OdError::Ok` on success. On error, `out` is zeroed.
 *
 * # Safety
 * - `handle` must be valid.
 * - `pixels_rgb` must point to at least `img_w * img_h * 3` bytes.
 * - `out` must be a valid pointer.
 */
enum OdError od_model_detect(struct ModelHandle *handle,
                             const uint8_t *pixels_rgb,
                             int32_t img_w,
                             int32_t img_h,
                             float conf_threshold,
                             float nms_threshold,
                             struct OdDetections *out);

/**
 * Free detection results.
 *
 * # Safety
 * `detections` must point to a valid `OdDetections` returned by `od_model_detect`.
 */
void od_detections_free(struct OdDetections *detections);

/**
 * Create a face pipeline (YuNet detector + ArcFace recognizer, ORT CPU).
 *
 * # Parameters
 * - `detector_path`: null-terminated path to YuNet `.onnx` file
 * - `recognizer_path`: null-terminated path to ArcFace `.onnx` file (e.g. `w600k_mbf.onnx`)
 *
 * # Returns
 * Opaque pointer, or null on error.
 *
 * # Safety
 * Both paths must be valid null-terminated C strings.
 */
struct FacePipelineHandle *face_pipeline_create(const char *detector_path,
                                                const char *recognizer_path);

/**
 * Create a face pipeline with CUDA acceleration.
 *
 * # Safety
 * Both paths must be valid null-terminated C strings.
 */
struct FacePipelineHandle *face_pipeline_create_cuda(const char *detector_path,
                                                     const char *recognizer_path);

/**
 * Create a face pipeline with TensorRT acceleration (via ORT).
 *
 * # Safety
 * Both paths must be valid null-terminated C strings.
 */
struct FacePipelineHandle *face_pipeline_create_tensorrt(const char *detector_path,
                                                         const char *recognizer_path);

/**
 * Returns the expected aligned face size (square side, read from the ONNX model).
 *
 * E.g. 112 for MobileFaceNet (w600k_mbf.onnx).
 * Go-side should call this instead of hardcoding a constant.
 *
 * # Safety
 * `handle` must be a valid pointer returned by `face_pipeline_create*`.
 */
uint32_t face_pipeline_aligned_size(const struct FacePipelineHandle *handle);

/**
 * Run face detection + recognition on an RGB image.
 *
 * # Parameters
 * - `handle`: face pipeline handle
 * - `pixels_rgb`: pointer to `width * height * 3` bytes (RGB, row-major, HWC)
 * - `img_w`, `img_h`: image dimensions in pixels
 * - `conf_threshold`: detection confidence threshold (e.g. 0.7)
 * - `nms_threshold`: NMS IoU threshold (e.g. 0.3)
 * - `out`: pointer to `FaceDetectionResults` struct, filled on success
 *
 * # Returns
 * `OdError::Ok` on success. On error, `out` is zeroed.
 *
 * # Safety
 * - `handle` must be valid.
 * - `pixels_rgb` must point to at least `img_w * img_h * 3` bytes.
 * - `out` must be a valid pointer.
 */
enum OdError face_pipeline_process(struct FacePipelineHandle *handle,
                                   const uint8_t *pixels_rgb,
                                   int32_t img_w,
                                   int32_t img_h,
                                   float conf_threshold,
                                   float nms_threshold,
                                   struct FaceDetectionResults *out);

/**
 * Extract embedding from a pre-aligned face image.
 *
 * The image must be aligned to the size returned by `face_pipeline_aligned_size()`
 * (typically 112x112).
 *
 * # Parameters
 * - `handle`: face pipeline handle
 * - `pixels_rgb`: pointer to aligned face RGB data (size x size x 3 bytes)
 * - `size`: aligned face size (e.g. 112)
 * - `out_embedding`: pointer to caller-allocated `[f32; 512]` buffer
 *
 * # Returns
 * `OdError::Ok` on success.
 *
 * # Safety
 * - `handle` must be valid.
 * - `pixels_rgb` must point to at least `size * size * 3` bytes.
 * - `out_embedding` must point to at least 512 f32 elements.
 */
enum OdError face_pipeline_embed(struct FacePipelineHandle *handle,
                                 const uint8_t *pixels_rgb,
                                 int32_t size,
                                 float *out_embedding);

/**
 * Free a face pipeline handle.
 *
 * # Safety
 * `handle` must have been returned by `face_pipeline_create*` and not yet freed.
 */
void face_pipeline_destroy(struct FacePipelineHandle *handle);

/**
 * Free face detection results.
 *
 * # Safety
 * `results` must point to a valid `FaceDetectionResults` returned by `face_pipeline_process`.
 */
void face_pipeline_results_free(struct FaceDetectionResults *results);

#endif  /* OD_BRIDGE_H */
