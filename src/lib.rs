use ndarray::{Array, Dimension};

mod activation;
mod layer;
mod loss;
mod model;
pub mod functions;

#[cfg(test)]
mod test_util;

pub use activation::Activation;
pub use layer::Layer;
pub use loss::CategoricalCrossEntorpy;
pub use model::{Model, Sequential};

pub fn clamp64<D: Dimension>(arr: &mut Array<f64, D>) {
    arr.map_inplace(|i| {
        if i.is_nan() {
            *i = 0.0
        } else if i.is_infinite() {
            *i = f64::MAX
        }
    });
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::activation::Activation;

    use super::{Layer, Model, Sequential};

    #[test]
    fn sequential_model_test() {
        let x_train = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 3.0, 3.0, 0.0,
                4.0, 4.0, 4.0, 0.0, 5.0, 5.0, 5.0, 0.0,
            ],
        )
        .unwrap();
        let y_train = Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0],
        )
        .unwrap();

        let mut model = Sequential::categorical();

        model.add_layer(Layer::dense(2, 10, Activation::relu()));
        // model.add_layer(Layer::dropout(10, 10, 0.9));
        model.add_layer(Layer::dense(10, 2, Activation::softmax()));

        model.train(&x_train, &y_train);

        // model.test(&x_train, &y_train);
    }
}
