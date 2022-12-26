use ndarray::Array2;

use crate::{CategoricalCrossEntorpy, Layer, Loss};

pub trait Model {
    fn fit(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>);

    fn predict(&self, inputs: &Array2<f64>) -> Array2<f64>;

    fn loss(&self, inputs: &Array2<f64>, targets: &Array2<f64>) -> f64;

    fn accuracy(&self, inputs: &Array2<f64>, targets: &Array2<f64>) -> f64;
}

/// Sequential model with default learning rate of 0.1
#[derive(Clone)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    loss_fn: Box<dyn Loss>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: vec![],
            loss_fn: CategoricalCrossEntorpy::new(),
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn set_loss_fn(&mut self, loss_fn: Box<dyn Loss>) {
        self.loss_fn = loss_fn;
    }
}

impl Model for Sequential {
    fn fit(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) {
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
        // backward
        let mut partial_deriv = self.loss_fn.backward(&last_output, labels);

        for l in self.layers.iter_mut().rev() {
            partial_deriv = l.backward(&partial_deriv);
        }
    }

    fn predict(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut last_output = inputs.clone();

        for l in self.layers.iter() {
            last_output = l.predict(last_output);
        }

        last_output
    }

    fn loss(&self, inputs: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        self.loss_fn.mean_loss(&self.predict(inputs), targets)
    }

    fn accuracy(&self, inputs: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        self.loss_fn.accuracy(&self.predict(inputs), targets)
    }
}
