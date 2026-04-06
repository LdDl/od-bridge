//! Smoke test: load a model, run detection on an image, print results.
//!
//! Usage:
//!   cargo run --release --example smoke_test -- \
//!     --model model.onnx --image test.jpg --width 416 --height 416
//!
//! Optional:
//!   --names classes.names    load class names from file (one per line)
//!   --conf  0.3              confidence threshold (default: 0.3)
//!   --nms   0.4              NMS IoU threshold (default: 0.4)

use std::ffi::CString;
use std::ptr;
use std::{env, fs, process};

use od_bridge::*;

struct Args {
    model: String,
    image: String,
    input_w: u32,
    input_h: u32,
    names: Option<String>,
    conf: f32,
    nms: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut a = Args {
        model: String::new(),
        image: String::new(),
        input_w: 416,
        input_h: 416,
        names: None,
        conf: 0.3,
        nms: 0.4,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; a.model = args[i].clone(); }
            "--image" => { i += 1; a.image = args[i].clone(); }
            "--width" => { i += 1; a.input_w = args[i].parse().expect("bad --width"); }
            "--height" => { i += 1; a.input_h = args[i].parse().expect("bad --height"); }
            "--names" => { i += 1; a.names = Some(args[i].clone()); }
            "--conf" => { i += 1; a.conf = args[i].parse().expect("bad --conf"); }
            "--nms" => { i += 1; a.nms = args[i].parse().expect("bad --nms"); }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: smoke_test --model MODEL --image IMAGE --width W --height H [--names FILE] [--conf F] [--nms F]");
                process::exit(1);
            }
        }
        i += 1;
    }

    if a.model.is_empty() || a.image.is_empty() {
        eprintln!("usage: smoke_test --model MODEL --image IMAGE --width W --height H [--names FILE] [--conf F] [--nms F]");
        process::exit(1);
    }
    a
}

fn load_names(path: &str) -> Vec<String> {
    fs::read_to_string(path)
        .expect("failed to read names file")
        .lines()
        .map(|s| s.to_string())
        .collect()
}

fn main() {
    let args = parse_args();

    let class_names = args.names.as_deref().map(load_names);

    let img = image::open(&args.image).unwrap_or_else(|e| {
        eprintln!("failed to open {}: {e}", args.image);
        process::exit(1);
    });
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as i32, rgb.height() as i32);
    let pixels = rgb.as_raw();
    println!("Image: {w}x{h}, {} bytes", pixels.len());

    let model_path = CString::new(args.model.as_str()).unwrap();
    let model = unsafe { od_model_create(model_path.as_ptr(), args.input_w, args.input_h) };
    if model.is_null() {
        eprintln!("failed to load model: {}", args.model);
        process::exit(1);
    }
    println!("Model loaded: {}", args.model);

    let mut out = OdDetections { data: ptr::null_mut(), len: 0 };
    let rc = unsafe {
        od_model_detect(model, pixels.as_ptr(), w, h, args.conf, args.nms, &mut out)
    };
    if rc != OdError::Ok {
        eprintln!("detection failed: {rc:?}");
        unsafe { od_model_free(model) };
        process::exit(1);
    }

    let count = out.len as usize;
    println!("Detections: {count}");

    if count > 0 {
        let detections = unsafe { std::slice::from_raw_parts(out.data, count) };
        for (i, d) in detections.iter().enumerate() {
            let class_label = class_names
                .as_ref()
                .and_then(|names| names.get(d.class_id as usize))
                .map(|s| s.as_str())
                .unwrap_or("?");
            println!(
                "  [{i}] class={class_label}({}) conf={:.1}% bbox=({}, {}, {}x{})",
                d.class_id,
                d.confidence * 100.0,
                d.bbox_x,
                d.bbox_y,
                d.bbox_w,
                d.bbox_h,
            );
        }
    }

    unsafe {
        od_detections_free(&mut out);
        od_model_free(model);
    }
    println!("Done.");
}
