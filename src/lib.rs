mod activation;
mod layer;
mod loss;
mod model;

pub use activation::Activation;
pub use layer::Layer;
pub use loss::CategoricalCrossEntorpy;
pub use model::{Model, Sequential};

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::activation::Activation;

    use super::{Layer, Model, Sequential};

    #[test]
    fn sequential_model() {
        // x_train = [[1.0, 2.0, 3.0],
        //            [4.0, 5.0, 6.0]]; (2 x 3)
        let x_train =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();

        // x_train = [[0.0, 1.0],
        //            [1.0, 0.0]]; (2 x 2)
        let y_train =
            Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut model = Sequential::categorical();

        model.add_layer(Layer::dense(3, 10, Activation::relu())); // in (2 x 3) -> out (2 x 10)
        model.add_layer(Layer::dropout(10, 10, 0.9)); // in (2 x 10) -> out (2 x 10)
        model.add_layer(Layer::dense(10, 2, Activation::softmax())); // in (2 x 10) -> out (2 x 2)
                                                                     // -> cce((2 x 2), (2 x 2)) -> (2 x 2)

        model.train(&x_train, &y_train);

        // activation2: inputs (2 x 2)                                      -> (2 x 2)
        // layer2:      weights (10 x 2) last_inputs (2 x 2) loss (2 x 2)   -> (2 x 2)
        // activation1: inputs (2 x 2)                                      -> (2 x 2)
        // layer1:      weights (3 x 10) last_inputs (2 x 10) loss (2 x 2)  -> (2 x 10)

        let x_test =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();

        let prediction = model.predict(&x_test);

        dbg!(prediction);
    }
}
