use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::backend::Backend,
};

//use rgb::RGB8;
//use textplots::{Chart, ColorPlot, Shape};

use crate::{
    dataset::{
        VitalSignsNews, VitalSignsNewsBatcher, VitalSignsNewsDataset, NUM_CLASSES, NUM_FEATURES,
    },
    model::{ModelConfig, ModelRecord},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let record: ModelRecord<B> = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    // parameters are features,classes,hidden size
    let model = ModelConfig::new(NUM_FEATURES, NUM_CLASSES, 256)
        .init(&device)
        .load_record(record);

    // Use a sample of 1000 items from the test split
    let dataset = VitalSignsNewsDataset::test();
    let items: Vec<VitalSignsNews> = dataset.iter().take(2000).collect();

    let batcher = VitalSignsNewsBatcher::new(device);
    let batch = batcher.batch(items.clone());
    let predicted = model.forward(batch.inputs.clone());
    let targets = batch.targets;

    let expected = targets.into_data().iter::<f32>().collect::<Vec<_>>();
    let predicted = predicted
        .iter_dim(0)
        .map(|item| item.into_data().into_vec::<f32>())
        .collect::<Vec<_>>();

    //let (index, value) = find_max_index(predicted[56].as_ref().unwrap());
    //println!("expected {}", expected[56]);
    //println!("predicted index {} value {} ", index, value);

    let mut count = 0;
    let mut correct = 0;
    for item in predicted.iter() {
        let (index, _value) = find_max_index(item.as_ref().unwrap());
        let fmt_expected: String = format!("{}", expected[count] as usize);
        let fmt_predicted: String = format!("{}", index);
        if fmt_expected.eq(&fmt_predicted) {
            correct += 1;
        } else {
            println!(
                "predicted {} {:?} expected {} ",
                count, predicted[count], index
            );
        }
        //println!(
        //    "item {} : predicted {} : expected : {}",
        //    count, fmt_predicted, fmt_expected
        //);
        count += 1;
    }
    println!(
        "summary total tests {} correct % {}",
        count,
        (correct as f32 / count as f32) * 100.0
    );

    /*
    let points = predicted
        .iter::<f32>()
        .zip(expected.iter::<f32>())
        .collect::<Vec<_>>();

    println!("Predicted vs. Expected Vital Signs Status");
    Chart::new_with_y_range(32, 32, 0., 5., 0., 5.)
        .linecolorplot(
            &Shape::Points(&points),
            RGB8 {
                r: 255,
                g: 85,
                b: 85,
            },
        )
        .display();

    // Print a single numeric value as an example
    println!("Predicted {:?} Expected {:?}", points[17].0, points[17].1);
    */
}

fn find_max_index(input: &Vec<f32>) -> (usize, f32) {
    let mut max_index = 0;
    let mut index = 0;
    let mut max = 0.0f32;
    for item in input.iter() {
        if item > &max {
            max_index = index;
            max = *item;
        }
        index += 1;
    }
    (max_index, max)
}
