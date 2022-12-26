use serde::{Deserialize, Serialize};

use ndarray::Array2;

use crate::functions::{
    accuracy, ccr_grad, ccr_mean, softmax, softmax_and_ccr_grad,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Loss {
    CategoricalCrossEntorpy(CategoricalCrossEntorpy),
    SoftmaxAndCategoricalCrossEntropy(SoftmaxAndCategoricalCrossEntropy),
}

impl Loss {
    pub fn new_cce() -> Self {
        Self::CategoricalCrossEntorpy(CategoricalCrossEntorpy)
    }

    pub fn new_scce() -> Self {
        Self::SoftmaxAndCategoricalCrossEntropy(
            SoftmaxAndCategoricalCrossEntropy,
        )
    }

    pub fn mean_loss(
        &self,
        predicions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        match self {
            Self::CategoricalCrossEntorpy(ccr) => {
                ccr.mean_loss(predicions, targets)
            }
            Self::SoftmaxAndCategoricalCrossEntropy(sccr) => {
                sccr.mean_loss(predicions, targets)
            }
        }
    }

    pub fn accuracy(
        &self,
        predicions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        match self {
            Self::CategoricalCrossEntorpy(ccr) => {
                ccr.accuracy(predicions, targets)
            }
            Self::SoftmaxAndCategoricalCrossEntropy(sccr) => {
                sccr.accuracy(predicions, targets)
            }
        }
    }

    pub fn backward(
        &self,
        dvalues: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        match self {
            Self::CategoricalCrossEntorpy(ccr) => {
                ccr.backward(dvalues, targets)
            }
            Self::SoftmaxAndCategoricalCrossEntropy(sccr) => {
                sccr.backward(dvalues, targets)
            }
        }
    }
}

trait LossFn {
    fn mean_loss(&self, predicions: &Array2<f64>, targets: &Array2<f64>)
        -> f64;

    fn accuracy(&self, predicions: &Array2<f64>, targets: &Array2<f64>) -> f64;

    fn backward(
        &self,
        dvalues: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64>;
}

/// Categorical cross entropy loss layer
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CategoricalCrossEntorpy;

impl LossFn for CategoricalCrossEntorpy {
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

/// Softmax activation and categorical cross entropy loss layer
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SoftmaxAndCategoricalCrossEntropy;

impl LossFn for SoftmaxAndCategoricalCrossEntropy {
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
