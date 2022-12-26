use dyn_clone::DynClone;
use ndarray::Array2;

use crate::functions::{
    accuracy, ccr_grad, ccr_mean, softmax, softmax_and_ccr_grad,
};

pub trait Loss: DynClone {
    fn mean_loss(&self, predicions: &Array2<f64>, targets: &Array2<f64>)
        -> f64;

    fn accuracy(&self, predicions: &Array2<f64>, targets: &Array2<f64>) -> f64;

    fn backward(
        &self,
        dvalues: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64>;
}

dyn_clone::clone_trait_object!(Loss);

/// Softmax activation and categorical cross entropy loss layer
#[derive(Clone)]
pub struct SoftmaxAndCategoricalCrossEntropy;

impl SoftmaxAndCategoricalCrossEntropy {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Loss for SoftmaxAndCategoricalCrossEntropy {
    fn mean_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        ccr_mean(&softmax(predictions), targets)
    }

    fn accuracy(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        accuracy(&softmax(predictions), targets)
    }

    fn backward(
        &self,
        loast_output: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        softmax_and_ccr_grad(&softmax(loast_output), targets)
    }
}

/// Categorical cross entropy loss layer
#[derive(Clone)]
pub struct CategoricalCrossEntorpy;

impl CategoricalCrossEntorpy {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Loss for CategoricalCrossEntorpy {
    fn mean_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        ccr_mean(predictions, targets)
    }

    fn accuracy(&self, predicions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        accuracy(predicions, targets)
    }

    fn backward(
        &self,
        dvalues: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        ccr_grad(&dvalues, targets)
    }
}
