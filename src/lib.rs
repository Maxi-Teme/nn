use ndarray::{Array, Dimension};

pub mod conversions;
pub mod functions;
pub mod helpers;
mod layer;
mod loss;
mod model;

pub use layer::{Dense, Dropout, Layer, ReLU, Softmax};
pub use loss::{
    CategoricalCrossEntorpy, Loss, SoftmaxAndCategoricalCrossEntropy,
};
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

    use crate::Layer;

    use super::{Model, Sequential};

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

        let mut model = Sequential::new();

        model.add_layer(Layer::new_dense(2, 10, None));
        model.add_layer(Layer::new_relu());
        model.add_layer(Layer::new_dense(10, 2, None));
        model.add_layer(Layer::new_sofmax());

        model.fit(&x_train, &y_train);

        // model.test(&x_train, &y_train);
    }
}
