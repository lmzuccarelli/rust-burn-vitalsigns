use crate::dataset::VitalSignsNewsBatch;
use burn::{
    nn::{loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
    dropout: Dropout,
}

#[derive(Config)]
pub struct ModelConfig {
    pub input_size: usize,
    pub classes: usize,
    pub hidden_size: usize,

    #[config(default = "0.5")]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_layer = LinearConfig::new(self.input_size, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, self.classes)
            .with_bias(true)
            .init(device);
        let d = DropoutConfig::new(self.dropout).init();

        Model {
            input_layer,
            output_layer,
            activation: Relu::new(),
            dropout: d,
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_layer.forward(input);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(inputs.clone());
        let loss = CrossEntropyLossConfig::new().init(&inputs.device());
        let loss = loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<VitalSignsNewsBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: VitalSignsNewsBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item.inputs, item.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<VitalSignsNewsBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: VitalSignsNewsBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.inputs, item.targets)
    }
}
