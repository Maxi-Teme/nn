use core::fmt;

use ndarray::{Array2, Axis, Ix2};

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
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.map(|x| x.max(0.0))
    }

    fn deriv(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.map(|x| if x <= &0.0 { 0.0 } else { 1.0 })
    }
}

#[derive(Debug, Default)]
pub struct Softmax {}

impl Softmax {
    /// from: https://docs.rs/drug/0.0.2/src/drug/lib.rs.html#77-89
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut softmax =
            inputs.to_owned().into_dimensionality::<Ix2>().unwrap();

        let max =
            softmax.fold_axis(
                Axis(1),
                0.0,
                |x, y| if *x > *y { *x } else { *y },
            );
        for ((b, _), x) in softmax.indexed_iter_mut() {
            *x = (*x - max[b]).exp();
        }
        let sum = softmax.sum_axis(Axis(1));
        for ((b, _), x) in softmax.indexed_iter_mut() {
            *x /= sum[b];
        }
        softmax
    }

    fn deriv(&self, inputs: &Array2<f64>) -> Array2<f64> {
        self.forward(&inputs).map(|x| x * (1.0 - x))
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use super::{ReLU, Softmax};

    #[test]
    fn relu_forward() {
        let relu = ReLU::default();

        let inputs = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.8, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let rel = relu.forward(&inputs);
        assert_eq!(rel.shape(), inputs.shape());
        assert_eq!(
            rel,
            Array2::from_shape_vec(
                (4, 2),
                vec![0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            )
            .unwrap()
        )
    }

    #[test]
    fn softmax_forward() {
        let softmax = Softmax::default();

        let inputs = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let soft = softmax.forward(&inputs);
        assert_eq!(soft.shape(), inputs.shape());

        let grad = softmax.deriv(&soft);
        assert_eq!(grad.shape(), inputs.shape());
    }
}
