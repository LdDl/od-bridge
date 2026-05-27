fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Compile glibc compat shims (see compat.c)
    cc::Build::new()
        .file("compat.c")
        .compile("compat");

    // Force compat symbols into the dynamic symbol table so ORT's PIC code
    // can resolve __isoc23_* at runtime on glibc < 2.38.
    // Rust's version script hides all non-Rust symbols under local: *, so we
    // append our own version script that explicitly globalises these three.
    let compat_map = std::path::PathBuf::from(&out_dir).join("compat.map");
    std::fs::write(&compat_map,
        "{ global: __isoc23_strtol; __isoc23_strtoll; __isoc23_strtoull; };\n"
    ).expect("write compat.map");
    println!("cargo:rustc-link-arg=-Wl,--version-script={}", compat_map.display());

    // Generate C header into crate root (for user convenience and installation)
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("OD_BRIDGE_H")
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file("od_bridge.h");

    // Generate pkg-config file from template into OUT_DIR only
    // (cargo publish does not allow build.rs to create files outside OUT_DIR)
    // For local use: cargo build writes to OUT_DIR, install scripts copy from there.
    // For convenience: `make install` or release workflow generates od_bridge.pc at install time.
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let prefix = std::env::var("OD_BRIDGE_PREFIX").unwrap_or_else(|_| "/usr/local".to_string());

    let template = std::fs::read_to_string(format!("{crate_dir}/od_bridge.pc.in"))
        .expect("Unable to read od_bridge.pc.in");
    let pc = template
        .replace("@PREFIX@", &prefix)
        .replace("@VERSION@", &version);
    std::fs::write(format!("{out_dir}/od_bridge.pc"), pc)
        .expect("Unable to write od_bridge.pc");
}
