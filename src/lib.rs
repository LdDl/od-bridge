//! od-bridge: C ABI bridge for od_opencv, designed for Go CGO integration.
//!
//! Provides opaque model handles and flat C structs for detection results.
//! Each model is independent: create one for plate detection, another for OCR.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use ndarray::Array3;
use od_opencv::model_factory::Model;
use od_opencv::model_trait::ObjectDetector;
use od_opencv::BBox;
use od_opencv::face_pipeline::FacePipeline;

/// Single detection result (flat, no pointers, safe for CGO memcpy).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OdDetection {
    /// Top-left corner X coordinate (pixels).
    pub bbox_x: i32,
    /// Top-left corner Y coordinate (pixels).
    pub bbox_y: i32,
    /// Bounding box width (pixels).
    pub bbox_w: i32,
    /// Bounding box height (pixels).
    pub bbox_h: i32,
    /// Predicted class index (zero-based).
    pub class_id: i32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
}

/// Detection results batch. Caller must free via `od_detections_free`.
#[repr(C)]
pub struct OdDetections {
    /// Pointer to the first element of the results array.
    pub data: *mut OdDetection,
    /// Number of detections in the array.
    pub len: i32,
}

/// Error code returned by all functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdError {
    /// No error.
    Ok = 0,
    /// Null pointer or invalid dimension passed.
    InvalidArgument = 1,
    /// ONNX model file could not be loaded.
    ModelLoadFailed = 2,
    /// Inference failed at runtime.
    DetectionFailed = 3,
    /// RGB pixel buffer could not be converted to Array3.
    ImageConvertFailed = 4,
}

/// Backend-specific model variant.
enum ModelInner {
    /// ONNX Runtime (CPU, CUDA, or TensorRT execution provider).
    Ort(od_opencv::backend_ort::ModelUltralyticsOrt),
    /// Native TensorRT engine.
    #[cfg(feature = "trt")]
    Trt(od_opencv::backend_tensorrt::ModelUltralyticsRt),
    /// Rockchip RKNN NPU.
    #[cfg(feature = "rknn")]
    Rknn(od_opencv::backend_rknn::ModelUltralyticsRknn),
}

/// Opaque model handle. Created by any `od_model_create_*` function.
pub struct ModelHandle {
    inner: ModelInner,
}

impl ModelHandle {
    /// Run detection on an `ImageBuffer`, dispatching to the active backend.
    fn detect(
        &mut self,
        img: &od_opencv::ImageBuffer,
        conf: f32,
        nms: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), OdError> {
        match &mut self.inner {
            ModelInner::Ort(m) => m.detect(img, conf, nms).map_err(|e| {
                eprintln!("od_model_detect (ort): {e:?}");
                OdError::DetectionFailed
            }),
            #[cfg(feature = "trt")]
            ModelInner::Trt(m) => m.detect(img, conf, nms).map_err(|e| {
                eprintln!("od_model_detect (trt): {e:?}");
                OdError::DetectionFailed
            }),
            #[cfg(feature = "rknn")]
            ModelInner::Rknn(m) => m.detect(img, conf, nms).map_err(|e| {
                eprintln!("od_model_detect (rknn): {e:?}");
                OdError::DetectionFailed
            }),
        }
    }
}

/// Helper: parse a C string pointer into a Rust `&str`.
/// Returns `None` if the pointer is null or not valid UTF-8.
unsafe fn parse_cstr(p: *const c_char) -> Option<&'static str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Helper: allocate a `ModelHandle` on the heap and return a raw pointer.
fn into_handle(inner: ModelInner) -> *mut ModelHandle {
    Box::into_raw(Box::new(ModelHandle { inner }))
}

/// Create a model from an ONNX file (ORT backend, CPU).
///
/// # Parameters
/// - `model_path`: null-terminated path to `.onnx` file
/// - `input_w`, `input_h`: model input dimensions (e.g. 416, 416)
///
/// # Returns
/// Opaque pointer, or null on error.
///
/// # Safety
/// `model_path` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_create(
    model_path: *const c_char,
    input_w: u32,
    input_h: u32,
) -> *mut ModelHandle {
    let Some(path) = (unsafe { parse_cstr(model_path) }) else {
        return ptr::null_mut();
    };
    match Model::ort(path, (input_w, input_h)) {
        Ok(model) => into_handle(ModelInner::Ort(model)),
        Err(e) => {
            eprintln!("od_model_create: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a model from an ONNX file with CUDA execution provider.
///
/// # Safety
/// `model_path` must be a valid null-terminated C string.
#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_create_cuda(
    model_path: *const c_char,
    input_w: u32,
    input_h: u32,
) -> *mut ModelHandle {
    let Some(path) = (unsafe { parse_cstr(model_path) }) else {
        return ptr::null_mut();
    };
    match Model::ort_cuda(path, (input_w, input_h)) {
        Ok(model) => into_handle(ModelInner::Ort(model)),
        Err(e) => {
            eprintln!("od_model_create_cuda: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a model from an ONNX file with TensorRT execution provider (via ORT).
///
/// # Safety
/// `model_path` must be a valid null-terminated C string.
#[cfg(feature = "tensorrt")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_create_tensorrt(
    model_path: *const c_char,
    input_w: u32,
    input_h: u32,
) -> *mut ModelHandle {
    let Some(path) = (unsafe { parse_cstr(model_path) }) else {
        return ptr::null_mut();
    };
    match Model::ort_tensorrt(path, (input_w, input_h)) {
        Ok(model) => into_handle(ModelInner::Ort(model)),
        Err(e) => {
            eprintln!("od_model_create_tensorrt: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a model from a serialized TensorRT engine file (native TensorRT, no ORT).
///
/// # Safety
/// `engine_path` must be a valid null-terminated C string.
#[cfg(feature = "trt")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_create_trt(
    engine_path: *const c_char,
) -> *mut ModelHandle {
    let Some(path) = (unsafe { parse_cstr(engine_path) }) else {
        return ptr::null_mut();
    };
    match Model::tensorrt(path) {
        Ok(model) => into_handle(ModelInner::Trt(model)),
        Err(e) => {
            eprintln!("od_model_create_trt: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a model from an RKNN model file (Rockchip NPU).
///
/// # Parameters
/// - `model_path`: null-terminated path to `.rknn` file
/// - `num_classes`: number of classes the model was trained on
///
/// # Safety
/// `model_path` must be a valid null-terminated C string.
#[cfg(feature = "rknn")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_create_rknn(
    model_path: *const c_char,
    num_classes: u32,
) -> *mut ModelHandle {
    let Some(path) = (unsafe { parse_cstr(model_path) }) else {
        return ptr::null_mut();
    };
    match Model::rknn(path, num_classes as usize) {
        Ok(model) => into_handle(ModelInner::Rknn(model)),
        Err(e) => {
            eprintln!("od_model_create_rknn: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Free a model handle.
///
/// # Safety
/// `handle` must have been returned by `od_model_create*` and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_free(handle: *mut ModelHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Run detection on an RGB image.
///
/// Works with any backend: the handle dispatches to the correct runtime internally.
///
/// # Parameters
/// - `handle`: model handle from any `od_model_create_*` function
/// - `pixels_rgb`: pointer to `width * height * 3` bytes (RGB, row-major, HWC)
/// - `img_w`, `img_h`: image dimensions in pixels
/// - `conf_threshold`: confidence threshold (e.g. 0.3)
/// - `nms_threshold`: NMS IoU threshold (e.g. 0.4)
/// - `out`: pointer to `OdDetections` struct, filled on success
///
/// # Returns
/// `OdError::Ok` on success. On error, `out` is zeroed.
///
/// # Safety
/// - `handle` must be valid.
/// - `pixels_rgb` must point to at least `img_w * img_h * 3` bytes.
/// - `out` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_model_detect(
    handle: *mut ModelHandle,
    pixels_rgb: *const u8,
    img_w: i32,
    img_h: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    out: *mut OdDetections,
) -> OdError {
    if handle.is_null() || pixels_rgb.is_null() || out.is_null() {
        return OdError::InvalidArgument;
    }
    if img_w <= 0 || img_h <= 0 {
        return OdError::InvalidArgument;
    }

    let model = unsafe { &mut *handle };
    let h = img_h as usize;
    let w = img_w as usize;
    let n_bytes = h * w * 3;

    let rgb_slice = unsafe { slice::from_raw_parts(pixels_rgb, n_bytes) };
    let arr = match Array3::from_shape_vec((h, w, 3), rgb_slice.to_vec()) {
        Ok(a) => a,
        Err(_) => {
            unsafe {
                (*out).data = ptr::null_mut();
                (*out).len = 0;
            }
            return OdError::ImageConvertFailed;
        }
    };

    let img_buf = od_opencv::ImageBuffer::from_rgb(arr);

    let (bboxes, class_ids, confidences) = match model.detect(&img_buf, conf_threshold, nms_threshold) {
        Ok(result) => result,
        Err(e) => {
            unsafe {
                (*out).data = ptr::null_mut();
                (*out).len = 0;
            }
            return e;
        }
    };

    let count = bboxes.len();
    if count == 0 {
        unsafe {
            (*out).data = ptr::null_mut();
            (*out).len = 0;
        }
        return OdError::Ok;
    }

    let mut results: Vec<OdDetection> = Vec::with_capacity(count);
    for i in 0..count {
        results.push(OdDetection {
            bbox_x: bboxes[i].x,
            bbox_y: bboxes[i].y,
            bbox_w: bboxes[i].width,
            bbox_h: bboxes[i].height,
            class_id: class_ids[i] as i32,
            confidence: confidences[i],
        });
    }

    let mut results = results.into_boxed_slice();
    unsafe {
        (*out).data = results.as_mut_ptr();
        (*out).len = count as i32;
    }
    std::mem::forget(results);

    OdError::Ok
}

/// Free detection results.
///
/// # Safety
/// `detections` must point to a valid `OdDetections` returned by `od_model_detect`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn od_detections_free(detections: *mut OdDetections) {
    if detections.is_null() {
        return;
    }
    let d = unsafe { &mut *detections };
    if !d.data.is_null() && d.len > 0 {
        let _ = unsafe {
            Vec::from_raw_parts(d.data, d.len as usize, d.len as usize)
        };
        d.data = ptr::null_mut();
        d.len = 0;
    }
}

/// Single face detection + recognition result (flat, safe for CGO memcpy).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaceDetectionResult {
    /// Bounding box top-left X (pixels).
    pub bbox_x: f32,
    /// Bounding box top-left Y (pixels).
    pub bbox_y: f32,
    /// Bounding box width (pixels).
    pub bbox_w: f32,
    /// Bounding box height (pixels).
    pub bbox_h: f32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
    /// 5 facial landmarks: [x0,y0, x1,y1, ..., x4,y4] (10 floats).
    pub landmarks: [f32; 10],
    /// 512-dimensional L2-normalized embedding.
    pub embedding: [f32; 512],
}

/// Face detection results batch. Caller must free via `face_pipeline_results_free`.
#[repr(C)]
pub struct FaceDetectionResults {
    /// Pointer to the first element of the results array.
    pub data: *mut FaceDetectionResult,
    /// Number of face detections in the array.
    pub len: i32,
}

/// Opaque face pipeline handle. Created by `face_pipeline_create*`.
pub struct FacePipelineHandle {
    inner: FacePipeline,
}

/// Create a face pipeline (YuNet detector + ArcFace recognizer, ORT CPU).
///
/// # Parameters
/// - `detector_path`: null-terminated path to YuNet `.onnx` file
/// - `recognizer_path`: null-terminated path to ArcFace `.onnx` file (e.g. `w600k_mbf.onnx`)
///
/// # Returns
/// Opaque pointer, or null on error.
///
/// # Safety
/// Both paths must be valid null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_create(
    detector_path: *const c_char,
    recognizer_path: *const c_char,
) -> *mut FacePipelineHandle {
    let Some(det_path) = (unsafe { parse_cstr(detector_path) }) else {
        return ptr::null_mut();
    };
    let Some(rec_path) = (unsafe { parse_cstr(recognizer_path) }) else {
        return ptr::null_mut();
    };
    match FacePipeline::new(det_path, rec_path) {
        Ok(pipeline) => Box::into_raw(Box::new(FacePipelineHandle { inner: pipeline })),
        Err(e) => {
            eprintln!("face_pipeline_create: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a face pipeline with CUDA acceleration.
///
/// # Safety
/// Both paths must be valid null-terminated C strings.
#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_create_cuda(
    detector_path: *const c_char,
    recognizer_path: *const c_char,
) -> *mut FacePipelineHandle {
    let Some(det_path) = (unsafe { parse_cstr(detector_path) }) else {
        return ptr::null_mut();
    };
    let Some(rec_path) = (unsafe { parse_cstr(recognizer_path) }) else {
        return ptr::null_mut();
    };
    match FacePipeline::new_cuda(det_path, rec_path) {
        Ok(pipeline) => Box::into_raw(Box::new(FacePipelineHandle { inner: pipeline })),
        Err(e) => {
            eprintln!("face_pipeline_create_cuda: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a face pipeline with TensorRT acceleration (via ORT).
///
/// # Safety
/// Both paths must be valid null-terminated C strings.
#[cfg(feature = "tensorrt")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_create_tensorrt(
    detector_path: *const c_char,
    recognizer_path: *const c_char,
) -> *mut FacePipelineHandle {
    let Some(det_path) = (unsafe { parse_cstr(detector_path) }) else {
        return ptr::null_mut();
    };
    let Some(rec_path) = (unsafe { parse_cstr(recognizer_path) }) else {
        return ptr::null_mut();
    };
    match FacePipeline::new_tensorrt(det_path, rec_path) {
        Ok(pipeline) => Box::into_raw(Box::new(FacePipelineHandle { inner: pipeline })),
        Err(e) => {
            eprintln!("face_pipeline_create_tensorrt: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Returns the expected aligned face size (square side, read from the ONNX model).
///
/// E.g. 112 for MobileFaceNet (w600k_mbf.onnx).
/// Go-side should call this instead of hardcoding a constant.
///
/// # Safety
/// `handle` must be a valid pointer returned by `face_pipeline_create*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_aligned_size(
    handle: *const FacePipelineHandle,
) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { &*handle }.inner.aligned_size()
}

/// Run face detection + recognition on an RGB image.
///
/// # Parameters
/// - `handle`: face pipeline handle
/// - `pixels_rgb`: pointer to `width * height * 3` bytes (RGB, row-major, HWC)
/// - `img_w`, `img_h`: image dimensions in pixels
/// - `conf_threshold`: detection confidence threshold (e.g. 0.7)
/// - `nms_threshold`: NMS IoU threshold (e.g. 0.3)
/// - `out`: pointer to `FaceDetectionResults` struct, filled on success
///
/// # Returns
/// `OdError::Ok` on success. On error, `out` is zeroed.
///
/// # Safety
/// - `handle` must be valid.
/// - `pixels_rgb` must point to at least `img_w * img_h * 3` bytes.
/// - `out` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_process(
    handle: *mut FacePipelineHandle,
    pixels_rgb: *const u8,
    img_w: i32,
    img_h: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    out: *mut FaceDetectionResults,
) -> OdError {
    if handle.is_null() || pixels_rgb.is_null() || out.is_null() {
        return OdError::InvalidArgument;
    }
    if img_w <= 0 || img_h <= 0 {
        return OdError::InvalidArgument;
    }

    let pipeline = unsafe { &mut *handle };
    let h = img_h as usize;
    let w = img_w as usize;
    let n_bytes = h * w * 3;

    let rgb_slice = unsafe { slice::from_raw_parts(pixels_rgb, n_bytes) };
    let arr = match Array3::from_shape_vec((h, w, 3), rgb_slice.to_vec()) {
        Ok(a) => a,
        Err(_) => {
            unsafe {
                (*out).data = ptr::null_mut();
                (*out).len = 0;
            }
            return OdError::ImageConvertFailed;
        }
    };

    let img_buf = od_opencv::ImageBuffer::from_rgb(arr);

    let faces = match pipeline.inner.process(&img_buf, conf_threshold, nms_threshold) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("face_pipeline_process: {e:?}");
            unsafe {
                (*out).data = ptr::null_mut();
                (*out).len = 0;
            }
            return OdError::DetectionFailed;
        }
    };

    let count = faces.len();
    if count == 0 {
        unsafe {
            (*out).data = ptr::null_mut();
            (*out).len = 0;
        }
        return OdError::Ok;
    }

    let mut results: Vec<FaceDetectionResult> = Vec::with_capacity(count);
    for face in &faces {
        let mut landmarks = [0.0f32; 10];
        for k in 0..5 {
            landmarks[k * 2] = face.landmarks[k][0];
            landmarks[k * 2 + 1] = face.landmarks[k][1];
        }
        results.push(FaceDetectionResult {
            bbox_x: face.x,
            bbox_y: face.y,
            bbox_w: face.width,
            bbox_h: face.height,
            confidence: face.confidence,
            landmarks,
            embedding: face.embedding,
        });
    }

    let mut results = results.into_boxed_slice();
    unsafe {
        (*out).data = results.as_mut_ptr();
        (*out).len = count as i32;
    }
    std::mem::forget(results);

    OdError::Ok
}

/// Extract embedding from a pre-aligned face image.
///
/// The image must be aligned to the size returned by `face_pipeline_aligned_size()`
/// (typically 112x112).
///
/// # Parameters
/// - `handle`: face pipeline handle
/// - `pixels_rgb`: pointer to aligned face RGB data (size x size x 3 bytes)
/// - `size`: aligned face size (e.g. 112)
/// - `out_embedding`: pointer to caller-allocated `[f32; 512]` buffer
///
/// # Returns
/// `OdError::Ok` on success.
///
/// # Safety
/// - `handle` must be valid.
/// - `pixels_rgb` must point to at least `size * size * 3` bytes.
/// - `out_embedding` must point to at least 512 f32 elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_embed(
    handle: *mut FacePipelineHandle,
    pixels_rgb: *const u8,
    size: i32,
    out_embedding: *mut f32,
) -> OdError {
    if handle.is_null() || pixels_rgb.is_null() || out_embedding.is_null() {
        return OdError::InvalidArgument;
    }
    if size <= 0 {
        return OdError::InvalidArgument;
    }

    let pipeline = unsafe { &mut *handle };
    let s = size as usize;
    let n_bytes = s * s * 3;

    let rgb_slice = unsafe { slice::from_raw_parts(pixels_rgb, n_bytes) };
    let arr = match Array3::from_shape_vec((s, s, 3), rgb_slice.to_vec()) {
        Ok(a) => a,
        Err(_) => return OdError::ImageConvertFailed,
    };

    let img_buf = od_opencv::ImageBuffer::from_rgb(arr);

    match pipeline.inner.embed(&img_buf) {
        Ok(embedding) => {
            unsafe {
                ptr::copy_nonoverlapping(embedding.as_ptr(), out_embedding, 512);
            }
            OdError::Ok
        }
        Err(e) => {
            eprintln!("face_pipeline_embed: {e:?}");
            OdError::DetectionFailed
        }
    }
}

/// Free a face pipeline handle.
///
/// # Safety
/// `handle` must have been returned by `face_pipeline_create*` and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_destroy(handle: *mut FacePipelineHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Free face detection results.
///
/// # Safety
/// `results` must point to a valid `FaceDetectionResults` returned by `face_pipeline_process`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn face_pipeline_results_free(results: *mut FaceDetectionResults) {
    if results.is_null() {
        return;
    }
    let r = unsafe { &mut *results };
    if !r.data.is_null() && r.len > 0 {
        let _ = unsafe {
            Vec::from_raw_parts(r.data, r.len as usize, r.len as usize)
        };
        r.data = ptr::null_mut();
        r.len = 0;
    }
}
