// Compile cuda/*.cu with nvcc (CUDA_HOME or nvcc on PATH) and link cudart + cuBLASLt.
// CLIPFAST_CUDA_ARCH: comma-separated SM versions (default "120a" = RTX PRO 6000 Blackwell; add "100a" for B200).
fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=CLIPFAST_CUDA_ARCH");
    let archs = std::env::var("CLIPFAST_CUDA_ARCH").unwrap_or_else(|_| "120a".into());
    let mut b = cc::Build::new();
    b.cuda(true).cudart("shared").debug(false).opt_level(3).flag("-O3").flag("-std=c++17").flag("-lineinfo")
        .flag("-Xptxas=-v,-warn-spills").warnings(false);
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    if let Ok(cut) = std::env::var("CUTLASS_DIR") {
        b.include(format!("{cut}/include")).define("CLIPFAST_CUTLASS", "1").flag("--expt-relaxed-constexpr");
    }
    let mut files: Vec<_> = std::fs::read_dir("cuda").unwrap().filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cu")).collect();
    files.sort();
    for f in &files {
        b.file(f);
    }
    let list: Vec<&str> = archs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for a in &list {
        b.flag(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    b.compile("clipfast_cuda");
    if let Some(home) = std::env::var_os("CUDA_HOME").or_else(|| std::env::var_os("CUDA_PATH")) {
        let home = std::path::PathBuf::from(home);
        for d in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            println!("cargo:rustc-link-search=native={}", home.join(d).display());
        }
    }
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
