use std::{env, path::PathBuf, process::Command};

fn compile_ptx(src: &str, out_name: &str, arch: &str) {
    println!("cargo:rerun-if-changed={src}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_path = out_dir.join(out_name);

    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "-O3",
            &format!("-arch={arch}"),
            src,
            "-o",
        ])
        .arg(&ptx_path)
        .status()
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("nvcc failed compiling {src} -> {out_name}");
    }

    println!("cargo:warning=Generated PTX at {}", ptx_path.display());
}

fn main() {
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

    compile_ptx("cuda/inc.cu", "inc.ptx", &arch);
    compile_ptx("cuda/degree.cu", "degree.ptx", &arch);
    compile_ptx("cuda/frontier_expand.cu", "frontier_expand.ptx", &arch);
}
