use core::fmt;

use ndarray::{Array2, Axis};

use crate::clamp64;

#[derive(Debug)]
pub enum Activation {
    ReLU(ReLU),
    Softmax(Softmax),
}

impl Activation {
    pub fn relu() -> Self {
        Self::ReLU(ReLU::default())
    }

    pub fn softmax() -> Self {
        Self::Softmax(Softmax::default())
    }

    pub fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::ReLU(relu) => relu.forward(inputs),
            Self::Softmax(softmax) => softmax.forward(inputs),
        }
    }

    pub fn deriv(&self, inputs: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::ReLU(relu) => relu.deriv(inputs),
            Self::Softmax(softmax) => softmax.deriv(inputs),
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReLU(_) => write!(f, "ReLU"),
            Self::Softmax(_) => write!(f, "Softmax"),
        }
    }
}

#[derive(Debug, Default)]
pub struct ReLU {}

impl ReLU {
    /// no shape change
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.map(|i| i.max(0.0))
    }

    /// no shape change
    fn deriv(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.map(|i| if *i < 0.0 { 0.0 } else { 1.0 })
    }
}

#[derive(Debug, Default)]
pub struct Softmax {}

impl Softmax {
    /// no shape change
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let max = inputs.fold(f64::EPSILON, |acc, x| x.max(acc));
        let mut exps = inputs.map(|i| (i - max).exp());

        exps.zip_mut_with(
            &inputs.sum_axis(Axis(1)).insert_axis(Axis(1)),
            |x, y| *x = *x / y,
        );

        clamp64(&mut exps);

        exps
    }

    /// no shape change
    fn deriv(&self, inputs: &Array2<f64>) -> Array2<f64> {
        // let exps = inputs.map(|i| i.exp());
        // let sums = inputs.sum_axis(Axis(1)).insert_axis(Axis(1));
        // let mut grad = &exps * (&sums - &exps) / sums.map(|i| i.powi(2));

        // clamp64(&mut grad);

        // grad
        self.forward(&inputs) - inputs
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::test_util;

    use super::{ReLU, Softmax};

    #[test]
    fn relu_forward() {
        let relu = ReLU::default();

        let inputs = Array2::from_shape_vec(
            (4, 2),
            vec![2.0, 0.8, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let rel = relu.forward(&inputs);
        assert_eq!(rel.shape(), inputs.shape());

        let inputs = test_util::get_random_data_array((2000, 10));
        let rel = relu.forward(&inputs);
        assert_eq!(rel.shape(), inputs.shape());
    }

    #[test]
    fn softmax_forward() {
        let softmax = Softmax::default();

        let inputs = Array2::from_shape_vec(
            (2, 4),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let soft = softmax.forward(&inputs);
        assert_eq!(soft.shape(), inputs.shape());

        let grad = softmax.deriv(&soft);
        assert_eq!(grad.shape(), inputs.shape());
    }
}
