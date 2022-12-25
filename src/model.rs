use std::fmt::Debug;

use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;

use crate::loss::Loss;
use crate::{CategoricalCrossEntorpy, Layer};

pub trait Model {
    fn train(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) -> f64;
}

pub struct Sequential<Lo: Loss + Debug> {
    layers: Vec<Layer>,
    loss_fn: Lo,
    learning_rate: f64,
}

impl<Lo: Loss + Debug> Sequential<Lo> {
    pub fn new(loss_fn: Lo, learning_rate: f64) -> Self {
        Self {
            layers: vec![],
            loss_fn,
            learning_rate,
        }
    }

    pub fn add_layer(&mut self, layer: Layer) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn set_loss_fn(&mut self, loss_fn: Lo) {
        self.loss_fn = loss_fn;
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn build(&mut self) -> Result<(), String> {
        todo!()
    }

    // pub fn test(&mut self, x_test: &Array2<f64>, y_test: &Array2<f64>) -> f64 {
    //     let mut input = x_test;

    //     for l in self.layers.iter_mut() {
    //         let (a, _) = l.forward(&input);
    //         input = a;
    //     }

    //     let predictions = self.argmax(&input);
    //     let y_test = self.argmax(y_test);

    //     let correct_classifications = eq(&predictions, &y_test, false);
    //     let number_of_correctly_classified =
    //         sum_all(&correct_classifications).0 as u64;

    //     number_of_correctly_classified as f64 / x_test.dims()[1] as f64
    // }

    fn _argmax(&self, predictions: &Array2<f64>) -> Array2<usize> {
        predictions
            .map_axis(Axis(1), |a| a.argmax_skipnan().unwrap())
            .insert_axis(Axis(1))
    }
}

impl Sequential<CategoricalCrossEntorpy> {
    /// categorical cross entropy, learning rate: 0.1
    pub fn categorical() -> Self {
        Self::new(CategoricalCrossEntorpy::default(), 0.1)
    }
}

impl<Lo: Loss + Debug> Model for Sequential<Lo> {
    fn train(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) -> f64 {
        assert!(
            self.layers.len() > 0,
            "Expected Sequential model to have at least one layer"
        );
        assert!(inputs.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        assert!(labels.iter().all(|i| !i.is_nan() && !i.is_infinite()));

        //
        // forward
        //
        let mut outs = Vec::with_capacity(self.layers.len());
        let mut input = inputs.clone();

        for l in self.layers.iter_mut() {
            let (a, z) = l.forward(&input);
            outs.push((input, z));
            input = a;
        }
        // outs.push((input, None));

        //
        // backward
        //
        let mean_loss = self.loss_fn.mean_loss(&input, labels);

        let mut partial_error = Array2::<f64>::ones(input.raw_dim());

        for (layer, (a, z)) in
            self.layers.iter_mut().rev().zip(outs.into_iter().rev())
        {
            partial_error =
                layer.backward(&partial_error, z, &a, self.learning_rate);
        }

        mean_loss
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::activation::Activation;
    use crate::{Layer, Model};

    use crate::test_util;

    use super::Sequential;

    #[test]
    fn sequential_model_test1() {
        let x_train =
            Array2::from_shape_vec((1, 784), test_util::ONE_IMAGE_X.to_vec())
                .unwrap();
        let y_train =
            Array2::from_shape_vec((1, 10), test_util::ONE_IMAGE_Y.to_vec())
                .unwrap();

        let mut model = Sequential::categorical();

        model.add_layer(Layer::dense(784, 400, Activation::relu()));
        model.add_layer(Layer::dropout(400, 1, 8));
        model.add_layer(Layer::dense(400, 400, Activation::relu()));
        model.add_layer(Layer::dropout(400, 1, 8));
        model.add_layer(Layer::dense(400, 10, Activation::softmax()));

        let loss = model.train(&x_train, &y_train);

        dbg!(&loss);

        let nann = f64::NAN;

        println!("{}", nann.clamp(f64::MIN, f64::MAX));
    }
}
