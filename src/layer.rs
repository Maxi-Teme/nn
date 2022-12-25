use ndarray::{Array1, Array2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::activation::Activation;
// use crate::clamp64;

pub enum Layer {
    Dense(Dense),
    Dropout(Dropout),
}

impl Layer {
    pub fn dense(n_inputs: usize, size: usize, activation: Activation) -> Self {
        Self::Dense(Dense::new(n_inputs, size, activation))
    }

    pub fn dropout(n_inputs: usize, numerator: u32, denominator: u32) -> Self {
        Self::Dropout(Dropout::new(n_inputs, numerator, denominator))
    }

    pub fn forward(
        &mut self,
        inputs: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        assert!(inputs.iter().all(|i| !i.is_nan()));

        match self {
            Self::Dense(dense) => dense.forward(inputs),
            Self::Dropout(dropout) => dropout.forward(inputs),
        }
    }

    pub fn backward(
        &mut self,
        partial_error: &Array2<f64>,
        z: Option<Array2<f64>>,
        a: &Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64> {
        // assert!(partial_error.iter().all(|i| !i.is_nan()));
        // assert!(a.iter().all(|i| !i.is_nan()));
        // assert!(!learning_rate.is_nan() && learning_rate > 0.0);

        match self {
            Self::Dense(dense) => {
                dense.backward(partial_error, z.unwrap(), a, learning_rate)
            }
            Self::Dropout(dropout) => dropout.backward(partial_error),
        }
    }
}

pub struct Dense {
    weights: Array2<f64>,
    biases: Array2<f64>,
    activation: Activation,
}

impl Dense {
    pub fn new(n_inputs: usize, size: usize, activation: Activation) -> Self {
        Self {
            biases: Array2::<f64>::random(
                (1, size as usize),
                Uniform::new(-0.2, 0.2),
            ),
            weights: Array2::<f64>::random(
                (n_inputs, size),
                Uniform::new(0.0, 1.0),
            ),
            activation,
        }
    }

    /// weights: (n_inputs, size)
    /// inputs:  (n_batch, n_inputs)
    /// return:  (n_batch, size)
    fn forward(
        &mut self,
        inputs: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        let out = inputs.dot(&self.weights) + &self.biases;
        // assert!(out.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        // let mut out = inputs.dot(&self.weights) + &self.biases;
        // clamp64(&mut out);

        let activation = self.activation.forward(&out);
        // assert!(activation.iter().all(|i| !i.is_nan() && !i.is_infinite()));

        return (activation, Some(out));
    }

    fn backward(
        &mut self,
        partial_error: &Array2<f64>,
        z: Array2<f64>,
        a: &Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64> {
        let error = self.activation.deriv(&z) * partial_error;
        // assert!(error.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        // let mut error = self.activation.deriv(&z) * partial_error;
        // clamp64(&mut error);

        let next_error = error.dot(&self.weights.t());
        // assert!(next_error.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        // let mut next_error = error.dot(&self.weights.t());
        // clamp64(&mut next_error);

        let eta = learning_rate / z.shape()[1] as f64;

        self.weights = &self.weights - a.t().dot(&error) * eta;
        // assert!(self.weights.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        // clamp64(&mut self.weights);

        self.biases = &self.biases - error * eta;
        // assert!(self.biases.iter().all(|i| !i.is_nan() && !i.is_infinite()));
        // clamp64(&mut self.biases);

        next_error
    }
}

pub struct Dropout {
    mask: Array1<f64>,
    frac: (u32, u32),
}

impl Dropout {
    pub fn new(n_inputs: usize, numerator: u32, denominator: u32) -> Dropout {
        Dropout {
            mask: Array1::<f64>::ones(n_inputs),
            frac: (numerator, denominator),
        }
    }

    fn update_mask(&self) -> Array1<f64> {
        let mut rng = ndarray_rand::rand::thread_rng();
        Array1::<f64>::ones(self.mask.len()).map(|i| {
            if rng.gen_ratio(self.frac.0, self.frac.1) {
                0f64
            } else {
                *i
            }
        })
    }

    pub fn forward(
        &mut self,
        z: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        self.update_mask();

        (z * &self.mask, None)
    }

    pub fn backward(&self, partial_error: &Array2<f64>) -> Array2<f64> {
        partial_error * &self.mask
    }
}

#[cfg(test)]
mod test {
    use crate::activation::Activation;

    use crate::test_util;

    use super::{Dense, Dropout};

    #[test]
    fn dropout_layer_forward_one_row_data() {
        let x_train = test_util::get_random_data_array((1, 27));

        let mut dropout1 = Dropout::new(27, 1, 8);

        let (out1, _) = dropout1.forward(&x_train);
        assert_eq!(out1.shape(), &[1, 27]);
    }

    #[test]
    fn dropout_layer_forward_multi_row_data() {
        let x_train = test_util::get_random_data_array((99, 27));

        let mut dropout1 = Dropout::new(1, 8, 27);

        let (out1, _) = dropout1.forward(&x_train);
        assert_eq!(out1.shape(), &[99, 27]);
    }

    #[test]
    fn dense_layer_forward_one_row_data() {
        let x_train = test_util::get_random_data_array((1, 3));

        let mut dense1 = Dense::new(3, 10, Activation::relu());

        let (out1, activated_out1) = dense1.forward(&x_train);
        assert_eq!(out1.shape(), &[1, 10]);
        assert_eq!(activated_out1.unwrap().shape(), &[1, 10]);

        let mut dense2 = Dense::new(10, 8, Activation::relu());

        let (out2, activated_out2) = dense2.forward(&out1);
        assert_eq!(out2.shape(), &[1, 8]);
        assert_eq!(activated_out2.unwrap().shape(), &[1, 8]);

        let mut dense3 = Dense::new(8, 200, Activation::softmax());

        let (out3, activated_out3) = dense3.forward(&out2);
        assert_eq!(out3.shape(), &[1, 200]);
        assert_eq!(activated_out3.unwrap().shape(), &[1, 200]);
    }

    #[test]
    fn dense_layer_forward_multi_row_data() {
        let x_train = test_util::get_random_data_array((3, 3));

        let mut dense1 = Dense::new(3, 10, Activation::relu());

        let (out1, activated_out1) = dense1.forward(&x_train);
        assert_eq!(out1.shape(), &[3, 10]);
        assert_eq!(activated_out1.unwrap().shape(), &[3, 10]);

        let mut dense2 = Dense::new(10, 8, Activation::relu());

        let (out2, activated_out2) = dense2.forward(&out1);
        assert_eq!(out2.shape(), &[3, 8]);
        assert_eq!(activated_out2.unwrap().shape(), &[3, 8]);

        let mut dense3 = Dense::new(8, 200, Activation::softmax());

        let (out3, activated_out3) = dense3.forward(&out2);
        assert_eq!(out3.shape(), &[3, 200]);
        assert_eq!(activated_out3.unwrap().shape(), &[3, 200]);
    }

    #[test]
    fn dense_layer_and_dropout_forward_one_row_data() {
        let x_train = test_util::get_random_data_array((1, 999));

        let mut dense1 = Dense::new(999, 10, Activation::relu());

        let (out1, activated_out1) = dense1.forward(&x_train);
        assert_eq!(out1.shape(), &[1, 10]);
        assert_eq!(activated_out1.unwrap().shape(), &[1, 10]);

        let mut dropout1 = Dropout::new(10, 1, 8);

        let (out1, _) = dropout1.forward(&x_train);
        assert_eq!(out1.shape(), &[1, 10]);

        let mut dense2 = Dense::new(10, 8, Activation::relu());

        let (out2, activated_out2) = dense2.forward(&out1);
        assert_eq!(out2.shape(), &[1, 8]);
        assert_eq!(activated_out2.unwrap().shape(), &[1, 8]);

        let mut dropout1 = Dropout::new(8, 1, 8);

        let (out1, _) = dropout1.forward(&x_train);
        assert_eq!(out1.shape(), &[1, 8]);
    }

    #[test]
    fn dense_layer_and_dropout_forward_multi_row_data() {
        let x_train = test_util::get_random_data_array((27, 999));

        let mut dense1 = Dense::new(999, 30, Activation::relu());

        let (out1, activated_out1) = dense1.forward(&x_train);
        assert_eq!(out1.shape(), &[27, 30]);
        assert_eq!(activated_out1.unwrap().shape(), &[27, 30]);

        let mut dropout1 = Dropout::new(30, 1, 8);

        let (out1, _) = dropout1.forward(&x_train);
        assert_eq!(out1.shape(), &[27, 30]);

        let mut dense2 = Dense::new(30, 8, Activation::relu());

        let (out2, activated_out2) = dense2.forward(&out1);
        assert_eq!(out2.shape(), &[27, 8]);
        assert_eq!(activated_out2.unwrap().shape(), &[27, 8]);

        let mut dropout1 = Dropout::new(8, 1, 8);

        let (out1, _) = dropout1.forward(&x_train);
        assert_eq!(out1.shape(), &[27, 8]);
    }
}
