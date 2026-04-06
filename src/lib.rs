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

/// Opaque model handle wrapping `ModelUltralyticsOrt`.
pub struct ModelHandle {
    /// Underlying ONNX Runtime model from od_opencv.
    inner: od_opencv::backend_ort::ModelUltralyticsOrt,
}

/// Create a model from an ONNX file (ORT backend, CPU).
///
/// # Parameters
/// - `model_path`: null-terminated path to `.onnx` file
/// - `input_w`, `input_h`: model input dimensions (e.g. 640, 640)
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
    if model_path.is_null() {
        return ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match Model::ort(path, (input_w, input_h)) {
        Ok(model) => Box::into_raw(Box::new(ModelHandle { inner: model })),
        Err(e) => {
            eprintln!("od_model_create: {e:?}");
            ptr::null_mut()
        }
    }
}

/// Create a model from an ONNX file with CUDA acceleration.
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
    if model_path.is_null() {
        return ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match Model::ort_cuda(path, (input_w, input_h)) {
        Ok(model) => Box::into_raw(Box::new(ModelHandle { inner: model })),
        Err(e) => {
            eprintln!("od_model_create_cuda: {e:?}");
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
/// # Parameters
/// - `handle`: model handle
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
    // Validate arguments
    if handle.is_null() || pixels_rgb.is_null() || out.is_null() {
        return OdError::InvalidArgument;
    }
    if img_w <= 0 || img_h <= 0 {
        return OdError::InvalidArgument;
    }

    let model = unsafe { &mut (*handle).inner };
    let h = img_h as usize;
    let w = img_w as usize;
    let n_bytes = h * w * 3;

    // Zero-copy view into Go's memory, then build Array3
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

    // Run detection
    let (bboxes, class_ids, confidences) =
        match model.detect(&img_buf, conf_threshold, nms_threshold) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("od_model_detect: {e:?}");
                unsafe {
                    (*out).data = ptr::null_mut();
                    (*out).len = 0;
                }
                return OdError::DetectionFailed;
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

    // Allocate results
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
