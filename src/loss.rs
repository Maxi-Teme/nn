use ndarray::Array2;

use crate::functions::{
    accuracy, ccr_grad, ccr_mean, softmax, softmax_and_ccr_grad,
};

pub trait Loss {
    fn mean_loss(
        &mut self,
        predicions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64;

    fn accuracy(
        &mut self,
        predicions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64;

    fn backward(&mut self, targets: &Array2<f64>) -> Array2<f64>;
}

/// Softmax activation and categorical cross entropy loss layer
pub struct SoftmaxAndCategoricalCrossEntropy {
    last_inputs: Array2<f64>,
}

impl SoftmaxAndCategoricalCrossEntropy {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            last_inputs: Array2::<f64>::zeros((0, 0)),
        })
    }
}

impl Loss for SoftmaxAndCategoricalCrossEntropy {
    fn mean_loss(
        &mut self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        self.last_inputs = softmax(predictions);

        ccr_mean(&self.last_inputs, targets)
    }

    fn accuracy(
        &mut self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        self.last_inputs = softmax(predictions);

        accuracy(&self.last_inputs, targets)
    }

    fn backward(&mut self, targets: &Array2<f64>) -> Array2<f64> {
        softmax_and_ccr_grad(&self.last_inputs, targets)
    }
}

/// Categorical cross entropy loss layer
pub struct CategoricalCrossEntorpy {
    last_inputs: Array2<f64>,
}

impl CategoricalCrossEntorpy {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            last_inputs: Array2::<f64>::zeros((0, 0)),
        })
    }
}

impl Loss for CategoricalCrossEntorpy {
    fn mean_loss(
        &mut self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        self.last_inputs = predictions.clone();
        ccr_mean(&self.last_inputs, targets)
    }

    fn accuracy(
        &mut self,
        predicions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        accuracy(predicions, targets)
    }

    fn backward(&mut self, targets: &Array2<f64>) -> Array2<f64> {
        ccr_grad(&self.last_inputs, targets)
    }
}
