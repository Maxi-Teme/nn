use std::fmt::Debug;

use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;

use crate::loss::Loss;
use crate::{CategoricalCrossEntorpy, Layer};

pub trait Model {
    fn info(&self, detail: bool);
    fn train(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) -> f64;
}

#[derive(Debug)]
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

    pub fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut input = x.clone();
        let ones = Array2::ones(x.raw_dim());

        for l in self.layers.iter_mut() {
            (input, _) = l.forward(&input, &ones);
        }

        input
    }

    pub fn test(
        &mut self,
        x_test: &Array2<f64>,
        y_test: &Array2<usize>,
    ) -> f64 {
        let x_size = x_test.shape()[0].clone() as f64;

        let predictions = self.predict(x_test);
        let predictions = self.argmax(predictions);
        let mut right = 0.0;

        for (p, y) in predictions
            .axis_iter(Axis(0))
            .zip(y_test.axis_iter(Axis(0)))
        {
            if p == y {
                right += 1.0;
            }
        }

        right / x_size
    }

    pub fn run_training(&mut self) {
        todo!()
    }

    fn argmax(&self, predictions: Array2<f64>) -> Array2<usize> {
        let mut preds = vec![];

        for p in predictions.axis_iter(Axis(0)) {
            preds.push(p.argmax().unwrap());
        }

        Array2::from_shape_vec((predictions.shape()[0], 1), preds).unwrap()
    }

    pub fn peek_weight(&self, layer: usize, index: (usize, usize)) {
        if let Layer::Dense(ref dense) = self.layers[layer] {
            dense.peek_weight(index);
        }
    }
}

impl Sequential<CategoricalCrossEntorpy> {
    /// categorical cross entropy, learning rate: 0.1
    pub fn categorical() -> Self {
        Self::new(CategoricalCrossEntorpy::default(), 0.1)
    }
}

impl<Lo: Loss + Debug> Model for Sequential<Lo> {
    fn info(&self, detail: bool) {
        println!("\n------------- model info start -------------");
        println!("Loss function: {:#?}", self.loss_fn);

        for layer in self.layers.iter() {
            layer.info(detail);
        }
        println!("------------- model info end -------------\n");
    }

    fn train(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) -> f64 {
        assert!(
            self.layers.len() > 0,
            "Expected Sequential model to have at least one layer"
        );

        //
        // forward
        //
        let ones = Array2::ones(inputs.raw_dim());
        let mut outs = Vec::with_capacity(self.layers.len());
        let mut input = inputs.clone();

        for l in self.layers.iter_mut() {
            let (a, z) = l.forward(&input, &ones);
            outs.push((input, z));
            input = a;
        }
        outs.push((input, None));

        //
        // backward
        //
        let mut out_iter = outs.into_iter().rev();
        let last_activation = &out_iter.next().unwrap().0;

        let mut partial_error =
            self.loss_fn.derivative(last_activation, labels);
        let loss = self.loss_fn.mean_loss(last_activation, labels);

        for (layer, (a, z)) in self.layers.iter_mut().rev().zip(out_iter) {
            partial_error = layer.backward(
                &partial_error,
                &z.unwrap(),
                &a,
                self.learning_rate,
            );
        }

        loss
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::activation::Activation;
    use crate::{Layer, Model};

    use super::Sequential;

    #[test]
    fn sequential_model_train_test() {
        let x_train = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let y_train = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();

        let mut model = Sequential::categorical();

        model.add_layer(Layer::dense(2, 10, Activation::relu()));
        model.add_layer(Layer::dense(10, 10, Activation::relu()));
        model.add_layer(Layer::dense(10, 2, Activation::softmax()));

        let mut loss = 0.0;
        for i in 0..10000 {
            if i % 1000 == 0 {
                println!("{}", loss);
                model.peek_weight(0, (0, 0));
                model.peek_weight(1, (0, 0));
                model.peek_weight(2, (0, 0));
            }

            loss = model.train(&x_train, &y_train);
        }

        let y_test = Array2::from_shape_vec((4, 1), vec![0, 1, 1, 1]).unwrap();

        let accuracy = model.test(&x_train, &y_test);
        println!("{:?}", accuracy);
    }
}
