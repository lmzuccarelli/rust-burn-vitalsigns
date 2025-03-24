use crate::dataset::{VitalSignsNewsBatcher, VitalSignsNewsDataset, NUM_CLASSES, NUM_FEATURES};
use crate::model::ModelConfig;
use burn::optim::AdamConfig;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::AccuracyMetric, metric::CpuMemory, metric::CpuTemperature, metric::CpuUse,
        metric::CudaMetric, metric::LossMetric, LearnerBuilder,
    },
};

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 20)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    #[config(default = 32)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let optimizer = AdamConfig::new();
    let config = ExpConfig::new(optimizer);
    let model = ModelConfig::new(NUM_FEATURES, NUM_CLASSES, 256).init(&device);
    B::seed(config.seed);

    // Define train/valid datasets and dataloaders
    let train_dataset = VitalSignsNewsDataset::train();
    let valid_dataset = VitalSignsNewsDataset::validation();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = VitalSignsNewsBatcher::<B>::new(device.clone());
    let batcher_validate = VitalSignsNewsBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_validate = DataLoaderBuilder::new(batcher_validate)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 1e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_validate);

    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
