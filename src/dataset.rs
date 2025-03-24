use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset, dataset::InMemDataset},
    prelude::*,
};

pub const NUM_FEATURES: usize = 9;
pub const NUM_CLASSES: usize = 3;

// Pre-computed statistics for the vitalsign dataset features
// Inputs are Age,Height,Weight,Sex,SPO2 RR,T,SBP,HR
const FEATURES_MIN: [f32; NUM_FEATURES] = [18.0, 1.5, 40.0, 0.0, 89.0, 7.0, 34.0, 70.0, 38.0];
const FEATURES_MAX: [f32; NUM_FEATURES] = [80.0, 1.9, 180.0, 1.0, 99.0, 26.0, 40.0, 230.0, 135.0];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VitalSignsNews {
    /// Age
    #[serde(rename = "Age")]
    pub age: f32,

    /// Height
    #[serde(rename = "Height")]
    pub height: f32,

    /// Weight
    #[serde(rename = "Weight")]
    pub weight: f32,

    /// Sex
    #[serde(rename = "Sex")]
    pub sex: f32,

    /// SPO2 - Oxygen Saturation
    #[serde(rename = "SPO2")]
    pub oxygen_saturation: f32,

    /// Respiratory Rate
    #[serde(rename = "RR")]
    pub respiratory_rate: f32,

    /// Temperature
    #[serde(rename = "T")]
    pub temperature: f32,

    /// Systolic Blood Pressure
    #[serde(rename = "SBP")]
    pub blood_pressure: f32,

    /// Heart Rate
    #[serde(rename = "HR")]
    pub heart_rate: f32,

    /// Status - the Label ;)
    #[serde(rename = "Status")]
    pub status: f32,
}

pub struct VitalSignsNewsDataset {
    dataset: InMemDataset<VitalSignsNews>,
}

impl VitalSignsNewsDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn validation() -> Self {
        Self::new("validation")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');
        let file_name = match split.as_ref() {
            "train" => "data/vital-signs-100.csv",
            "validation" => "data/vital-signs-validate.csv",
            "test" => "data/vital-signs-test.csv",
            _ => "data/vital-signs-100.csv",
        };
        let dataset = InMemDataset::from_csv(file_name, &rdr).unwrap();
        Self { dataset }
    }
}

// must implement get and len
impl Dataset<VitalSignsNews> for VitalSignsNewsDataset {
    fn get(&self, index: usize) -> Option<VitalSignsNews> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Normalizer for the vital signs dataset.
#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub min: Tensor<B, 2>,
    pub max: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    /// Normalizes the input according to the vital signs data set min/max.
    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }
}

#[derive(Clone, Debug)]
pub struct VitalSignsNewsBatcher<B: Backend> {
    device: B::Device,
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct VitalSignsNewsBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> VitalSignsNewsBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device.clone(),
            normalizer: Normalizer::new(&device, &FEATURES_MIN, &FEATURES_MAX),
        }
    }
}

impl<B: Backend> Batcher<VitalSignsNews, VitalSignsNewsBatch<B>> for VitalSignsNewsBatcher<B> {
    fn batch(&self, items: Vec<VitalSignsNews>) -> VitalSignsNewsBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.age,
                    item.height,
                    item.weight,
                    item.sex,
                    item.oxygen_saturation,
                    item.respiratory_rate,
                    item.temperature,
                    item.blood_pressure,
                    item.heart_rate,
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.normalizer.normalize(inputs);
        let mut targets: Vec<Tensor<B, 1, Int>> = Vec::new();
        for item in items.iter() {
            let target_tensor = Tensor::<B, 1, Int>::from_data([item.status], &self.device);
            targets.push(target_tensor.unsqueeze());
        }

        let targets = Tensor::cat(targets, 0);
        VitalSignsNewsBatch { inputs, targets }
    }
}
