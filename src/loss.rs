use ndarray::{Array2, Axis, Zip};

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
    /// − ∑ y ln(p)
    fn mean_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> f64 {
        Zip::from(
            predictions
                .map(|i| i.clamp(f64::EPSILON, 1.0 - f64::EPSILON).ln())
                .axis_iter(Axis(0)),
        )
        .and(targets.axis_iter(Axis(0)))
        .map_collect(|p, t| -p.dot(&t))
        .mean()
        .unwrap()
    }

    fn derivative(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        (-1.0 * targets) / predictions + (1.0 - targets) / (1.0 - predictions)
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use super::{CategoricalCrossEntorpy, Loss};

    #[test]
    fn calculate_categorical_cross_entropy() {
        let ccr = CategoricalCrossEntorpy::default();

        let predicions = Array2::from_shape_vec(
            (3, 2),
            vec![0.7, 0.3, 0.5, 0.5, 0.05, 0.95],
        )
        .unwrap();
        let targets =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                .unwrap();

        let loss = ccr.mean_loss(&predicions, &targets);
        let loss = Array2::from_elem(targets.raw_dim(), loss);

        assert!(
            loss.get((0, 0)).unwrap() < &1.0,
            "loss was grater than 1.0. Got: {}",
            loss.get((0, 0)).unwrap()
        );

        let predicions = Array2::from_shape_vec(
            (3, 2),
            vec![0.99, 0.0, 0.99, 0.0, 0.0, 0.99],
        )
        .unwrap();
        let targets =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                .unwrap();

        let loss = ccr.mean_loss(&predicions, &targets);
        let loss = Array2::from_elem(targets.raw_dim(), loss);

        assert!(
            loss.get((0, 0)).unwrap() < &0.02,
            "loss was grater than 0.02. Got: {}",
            loss.get((0, 0)).unwrap()
        );

        let predicions = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.99, 0.0, 0.99, 0.99, 0.0],
        )
        .unwrap();
        let targets =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                .unwrap();

        let loss = ccr.mean_loss(&predicions, &targets);
        let loss = Array2::from_elem(targets.raw_dim(), loss);

        assert!(
            loss.get((0, 0)).unwrap() > &35.0,
            "loss was less than 35.0. Got: {}",
            loss.get((0, 0)).unwrap()
        );
    }
}
