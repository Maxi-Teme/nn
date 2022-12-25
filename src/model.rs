use ndarray::Array2;

use crate::{CategoricalCrossEntorpy, Layer, Loss};

pub trait Model {
    fn train(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>);

    fn predict(&mut self, inputs: &Array2<f64>) -> Array2<f64>;
}

/// Sequential model with default learning rate of 0.1
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    loss_fn: Box<dyn Loss>,

    last_loss: f64,
    last_accuracy: f64,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: vec![],
            loss_fn: CategoricalCrossEntorpy::new(),

            last_loss: 0.0,
            last_accuracy: 0.0,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn set_loss_fn(&mut self, loss_fn: Box<dyn Loss>) {
        self.loss_fn = loss_fn;
    }

    pub fn loss(&self) -> f64 {
        self.last_loss
    }

    pub fn accuracy(&self) -> f64 {
        self.last_accuracy
    }
}

impl Model for Sequential {
    fn train(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) {
        assert!(
            self.layers.len() > 0,
            "Expected Sequential model to have at least one layer"
        );
        assert!(inputs.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        assert!(labels.iter().all(|i| !i.is_nan() && !i.is_infinite()));

        //
        // forward
        let mut last_output = inputs.clone();

        for l in self.layers.iter_mut() {
            last_output = l.forward(last_output);
        }

        //
        // update loss and accuracy
        self.last_loss = self.loss_fn.mean_loss(&last_output, labels);
        self.last_accuracy = self.loss_fn.accuracy(&last_output, labels);

        //
        // backward
        let mut partial_deriv = self.loss_fn.backward(labels);

        for l in self.layers.iter_mut().rev() {
            partial_deriv = l.backward(&partial_deriv);
        }
    }

    fn predict(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut last_output = inputs.clone();

        for l in self.layers.iter() {
            last_output = l.predict(last_output);
        }

        last_output
    }
}
