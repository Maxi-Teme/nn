use ndarray::{Array2, Axis};

use crate::clamp64;

pub trait Loss {
    fn mean_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64;

    fn derivative(&self, y: &Array2<f64>, a: &Array2<f64>) -> Array2<f64>;
}

#[derive(Debug, Default)]
pub struct CategoricalCrossEntorpy;

impl Loss for CategoricalCrossEntorpy {
    /// − ∑ y * ln(p)
    fn mean_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        // assert_eq!(predictions.shape(), targets.shape());
        // assert!(predictions.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        // assert!(targets.iter().all(|i| !i.is_nan() && !i.is_infinite()));

        -(predictions.map(|p| (p + f64::EPSILON).ln()) * targets)
            .mean_axis(Axis(0))
            .unwrap()
            .sum()
    }

    /// (-1.0 * targets) / predictions + (1.0 - targets) / (1.0 - predictions)
    fn derivative(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        assert!(predictions.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        assert!(targets.iter().all(|i| !i.is_nan() && !i.is_infinite()));

        let mut grad = (-1.0 * targets) / predictions
            + (1.0 - targets) / (1.0 - predictions);

        clamp64(&mut grad);

        grad
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array2, Axis};

    use crate::test_util;

    use super::{CategoricalCrossEntorpy, Loss};

    #[test]
    fn calculate_categorical_cross_entropy_forward() {
        let ccr = CategoricalCrossEntorpy::default();

        let predicions =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();
        let targets =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();
        let loss = ccr.mean_loss(&predicions, &targets);
        assert!(loss < 10e-10, "loss was grater than 1.0. Got: {}", loss);

        let targets =
            Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
                .unwrap();
        let loss = ccr.mean_loss(&predicions, &targets);
        assert!(loss > 35.0, "loss was grater than 1.0. Got: {}", loss);
        dbg!(&loss);
    }

    fn mean_loss1(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        -(predictions.map(|p| (p + f64::EPSILON).ln()) * targets).sum()
            / predictions.shape()[0] as f64
    }

    fn mean_loss2(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        -(predictions.map(|p| (p + f64::EPSILON).ln()) * targets)
            .mean_axis(Axis(0))
            .unwrap()
            .sum()
    }

    #[test]
    fn test_ccr() {
        let predicions =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();
        let targets =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .unwrap();
        let loss1 = mean_loss1(&predicions, &targets);
        let loss2 = mean_loss2(&predicions, &targets);
        dbg!(&loss1, &loss2);
        assert_eq!(loss1, loss2);
    }
}
