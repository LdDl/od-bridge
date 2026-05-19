//! YuNet face detection via od-bridge FFI (detection only, no embeddings).
//!
//! Demonstrates face_pipeline_process on a multi-face image.
//! Uses oscar_selfies.jpg as the test image.
//!
//! Usage:
//!   cargo run --release --example face_yunet -- \
//!     --detector face_detection_yunet_2023mar.onnx \
//!     --recognizer w600k_mbf.onnx \
//!     --image oscar_selfies.jpg
//!
//! Optional:
//!   --conf  0.5   confidence threshold (default: 0.5)
//!   --nms   0.3   NMS IoU threshold (default: 0.3)

use std::ffi::CString;
use std::ptr;
use std::{env, process};

use od_bridge::*;

struct Args {
    detector: String,
    recognizer: String,
    image: String,
    conf: f32,
    nms: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut a = Args {
        detector: String::new(),
        recognizer: String::new(),
        image: String::new(),
        conf: 0.5,
        nms: 0.3,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--detector" => { i += 1; a.detector = args[i].clone(); }
            "--recognizer" => { i += 1; a.recognizer = args[i].clone(); }
            "--image" => { i += 1; a.image = args[i].clone(); }
            "--conf" => { i += 1; a.conf = args[i].parse().expect("bad --conf"); }
            "--nms" => { i += 1; a.nms = args[i].parse().expect("bad --nms"); }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: face_yunet --detector DET.onnx --recognizer REC.onnx --image IMG [--conf F] [--nms F]");
                process::exit(1);
            }
        }
        i += 1;
    }

    if a.detector.is_empty() || a.recognizer.is_empty() || a.image.is_empty() {
        eprintln!("usage: face_yunet --detector DET.onnx --recognizer REC.onnx --image IMG [--conf F] [--nms F]");
        process::exit(1);
    }
    a
}

fn main() {
    let args = parse_args();

    let img = image::open(&args.image).unwrap_or_else(|e| {
        eprintln!("failed to open {}: {e}", args.image);
        process::exit(1);
    });
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as i32, rgb.height() as i32);
    let pixels = rgb.as_raw();
    println!("Image: {w}x{h} ({} bytes)", pixels.len());

    let det_path = CString::new(args.detector.as_str()).unwrap();
    let rec_path = CString::new(args.recognizer.as_str()).unwrap();
    let pipeline = unsafe { face_pipeline_create(det_path.as_ptr(), rec_path.as_ptr()) };
    if pipeline.is_null() {
        eprintln!("failed to create face pipeline");
        process::exit(1);
    }
    println!("Pipeline created");

    let mut out = FaceDetectionResults { data: ptr::null_mut(), len: 0 };
    let rc = unsafe {
        face_pipeline_process(pipeline, pixels.as_ptr(), w, h, args.conf, args.nms, &mut out)
    };
    if rc != OdError::Ok {
        eprintln!("face pipeline failed: {rc:?}");
        unsafe { face_pipeline_destroy(pipeline) };
        process::exit(1);
    }

    let count = out.len as usize;
    println!("Faces detected: {count}");

    if count > 0 {
        let faces = unsafe { std::slice::from_raw_parts(out.data, count) };
        for (i, f) in faces.iter().enumerate() {
            println!(
                "  [{i}] conf={:.1}% bbox=({:.1}, {:.1}, {:.1}x{:.1})",
                f.confidence * 100.0,
                f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
            );
            println!(
                "       landmarks: ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1})",
                f.landmarks[0], f.landmarks[1],
                f.landmarks[2], f.landmarks[3],
                f.landmarks[4], f.landmarks[5],
                f.landmarks[6], f.landmarks[7],
                f.landmarks[8], f.landmarks[9],
            );
        }
    }

    unsafe {
        face_pipeline_results_free(&mut out);
        face_pipeline_destroy(pipeline);
    }
    println!("Done.");
}
