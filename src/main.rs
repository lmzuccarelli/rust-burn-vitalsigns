use burn_autodiff::Autodiff;
use burn_cuda::{Cuda, CudaDevice};
use std::env;

mod dataset;
mod inference;
mod model;
mod training;

fn main() {
    let device = CudaDevice::default();
    let artifact_dir = "vital-signs";
    let args: Vec<String> = env::args().collect();
    type MyBackend = Cuda<f32, i32>;
    if args.len() > 1 {
        if args[1] == "train" {
            type MyAutodiffBackend = Autodiff<MyBackend>;
            training::run::<MyAutodiffBackend>(artifact_dir, device.clone());
        } else {
            inference::infer::<MyBackend>(artifact_dir, device);
        }
    }
}
