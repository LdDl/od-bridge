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
