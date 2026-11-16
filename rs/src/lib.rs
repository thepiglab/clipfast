//! clipfast-rs: BiomedCLIP inference in Rust + CUDA.
#[cfg(target_arch = "x86_64")]
pub mod cpu;
pub mod engine;
pub mod export;
pub mod frontend;
pub mod prep;
pub mod search;
pub mod server;
#[cfg(feature = "python")]
mod py;
