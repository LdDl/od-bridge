//! Benchmark: measure inference throughput.
//!
//! Usage:
//!   cargo run --release --example bench -- \
//!     --model model.onnx --image test.jpg --width 416 --height 416
//!
//! Optional:
//!   --conf    0.3    confidence threshold (default: 0.3)
//!   --nms     0.4    NMS IoU threshold (default: 0.4)
//!   --warmup  3      warmup iterations (default: 3)
//!   --iters   50     benchmark iterations (default: 50)

use std::ffi::CString;
use std::ptr;
use std::time::Instant;
use std::{env, process};

use od_bridge::*;

struct Args {
    model: String,
    image: String,
    input_w: u32,
    input_h: u32,
    conf: f32,
    nms: f32,
    warmup: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut a = Args {
        model: String::new(),
        image: String::new(),
        input_w: 416,
        input_h: 416,
        conf: 0.3,
        nms: 0.4,
        warmup: 3,
        iters: 50,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; a.model = args[i].clone(); }
            "--image" => { i += 1; a.image = args[i].clone(); }
            "--width" => { i += 1; a.input_w = args[i].parse().expect("bad --width"); }
            "--height" => { i += 1; a.input_h = args[i].parse().expect("bad --height"); }
            "--conf" => { i += 1; a.conf = args[i].parse().expect("bad --conf"); }
            "--nms" => { i += 1; a.nms = args[i].parse().expect("bad --nms"); }
            "--warmup" => { i += 1; a.warmup = args[i].parse().expect("bad --warmup"); }
            "--iters" => { i += 1; a.iters = args[i].parse().expect("bad --iters"); }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: bench --model MODEL --image IMAGE --width W --height H [--conf F] [--nms F] [--warmup N] [--iters N]");
                process::exit(1);
            }
        }
        i += 1;
    }

    if a.model.is_empty() || a.image.is_empty() {
        eprintln!("usage: bench --model MODEL --image IMAGE --width W --height H [--conf F] [--nms F] [--warmup N] [--iters N]");
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

    let model_path = CString::new(args.model.as_str()).unwrap();
    let model = unsafe { od_model_create(model_path.as_ptr(), args.input_w, args.input_h) };
    if model.is_null() {
        eprintln!("failed to load model: {}", args.model);
        process::exit(1);
    }

    println!(
        "Model: {}\nImage: {w}x{h}\nWarmup: {} iters\nBenchmark: {} iters\n",
        args.model, args.warmup, args.iters
    );

    // Warmup
    for _ in 0..args.warmup {
        let mut out = OdDetections { data: ptr::null_mut(), len: 0 };
        unsafe {
            od_model_detect(model, pixels.as_ptr(), w, h, args.conf, args.nms, &mut out);
            od_detections_free(&mut out);
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..args.iters {
        let mut out = OdDetections { data: ptr::null_mut(), len: 0 };
        unsafe {
            od_model_detect(model, pixels.as_ptr(), w, h, args.conf, args.nms, &mut out);
            od_detections_free(&mut out);
        }
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() / args.iters as f64 * 1000.0;
    let fps = args.iters as f64 / elapsed.as_secs_f64();

    println!(
        "{} iters in {:.2?}\navg = {:.2} ms/frame\n{:.1} FPS",
        args.iters, elapsed, avg_ms, fps
    );

    unsafe { od_model_free(model) };
}
