use std::ops::Mul;

use ndarray::{Array2, Axis};
use ndarray_rand::rand;
use ndarray_rand::rand_distr::{Bernoulli, Uniform};
use ndarray_rand::RandomExt;

use crate::activation::Activation;

#[derive(Debug)]
pub enum Layer {
    Dense(Dense),
    DropOut(DropOut),
}

impl Layer {
    pub fn dense(n_inputs: usize, size: usize, activation: Activation) -> Self {
        Self::Dense(Dense::new(n_inputs, size, activation))
    }

    pub fn dropout(n_inputs: usize, size: usize, p: f64) -> Self {
        Self::DropOut(DropOut::new(n_inputs, size, p))
    }

    pub fn forward(
        &mut self,
        inputs: &Array2<f64>,
        ones: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        match self {
            Self::Dense(dense) => dense.forward(inputs, ones),
            Self::DropOut(dropout) => dropout.forward(inputs),
        }
    }

    pub fn backward(
        &mut self,
        partial_error: &Array2<f64>,
        z: &Array2<f64>,
        a: &Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64> {
        match self {
            Self::Dense(dense) => {
                dense.backward(partial_error, z, a, learning_rate)
            }
            Self::DropOut(dropout) => dropout.backward(partial_error),
        }
    }

    pub fn info(&self, detail: bool) {
        match self {
            Self::Dense(dense) => dense.info(detail),
            Self::DropOut(dropout) => dropout.info(detail),
        }
    }
}

#[derive(Debug)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array2<f64>,
    activation: Activation,
}

impl Dense {
    pub fn new(n_inputs: usize, size: usize, activation: Activation) -> Self {
        Self {
            weights: Array2::random((n_inputs, size), Uniform::new(-0.2, 0.2)),
            biases: Array2::zeros((1, size)),
            activation,
        }
    }

    fn forward(
        &mut self,
        inputs: &Array2<f64>,
        ones: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        // let weighted_inputs = self.weights.dot(inputs);
        // let bias_matrix = self.biases.dot(ones);

        let z = inputs.dot(&self.weights) + &self.biases;
        let a = self.activation.forward(&z);

        (a, Some(z))
    }

    fn backward(
        &mut self,
        partial_error: &Array2<f64>,
        z: &Array2<f64>,
        a: &Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64> {
        let error = self.activation.deriv(z) * partial_error;
        let next_partial_error = error.dot(&self.weights.t());
        
        let eta = learning_rate / z.shape()[0] as f64;
        
        let weights_error = &error.t().dot(a);
        self.weights = &self.weights - &weights_error.t() * eta;

        let bias_error = partial_error.sum_axis(Axis(0));
        self.biases = &self.biases - bias_error * eta;
        
        next_partial_error
    }

    pub fn shape_weights(&self) -> &[usize] {
        self.weights.shape()
    }

    pub fn info(&self, detail: bool) {
        if detail {
            println!("{:#?}", self);
        } else {
            println!(
                "Dense Layer: weights shape: {:?}, biases shape: {:?}, \
activation: {:?}",
                self.weights.shape(),
                self.biases.shape(),
                self.activation,
            );
        }
    }

    pub fn peek_weight(&self, index: (usize, usize)) {
        println!(
            "weight at [{},{}]: {}",
            index.0,
            index.1,
            self.weights.get(index).unwrap()
        );
    }
}

#[derive(Debug)]
pub struct DropOut {
    distribution: Bernoulli,
    last_dropout: Array2<f64>,
}

impl DropOut {
    pub fn new(n_inputs: usize, size: usize, p: f64) -> Self {
        assert!(0.0 <= p && p <= 1.0, "p must be between 0.0 and 1.0");

        Self {
            distribution: Bernoulli::new(p).unwrap(),
            last_dropout: Array2::zeros((n_inputs, size)),
        }
    }

    fn forward(
        &mut self,
        inputs: &Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        self.last_dropout = Array2::random_using(
            inputs.raw_dim(),
            self.distribution,
            &mut rand::thread_rng(),
        )
        .map(|b| if *b { 1.0 } else { 0.0 });

        (inputs.mul(&self.last_dropout), None)
    }

    fn backward(&mut self, partial_error: &Array2<f64>) -> Array2<f64> {
        partial_error.dot(&self.last_dropout)
    }

    pub fn shape_dropout(&self) -> &[usize] {
        self.last_dropout.shape()
    }

    pub fn info(&self, detail: bool) {
        if detail {
            println!("{:#?}", self);
        } else {
            println!("Dense Dropout");
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::activation::Activation;

    use super::{Dense, DropOut};

    #[test]
    fn dense_layer_forward() {
        let mut dense = Dense::new(3, 10, Activation::relu());

        let x_train =
            Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let ones = Array2::ones(x_train.raw_dim());
        let (_, forward) = dense.forward(&x_train, &ones);
        assert_eq!(forward.unwrap().shape(), &[1, 10]);

        let x_train =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let (_, forward) = dense.forward(&x_train, &ones);
        assert_eq!(forward.unwrap().shape(), &[2, 10]);

        let x_train = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
        )
        .unwrap();
        let (_, forward) = dense.forward(&x_train, &ones);
        assert_eq!(forward.unwrap().shape(), &[10, 10]);
    }

    #[test]
    fn dropout_layer_forward() {
        let mut dropout = DropOut::new(3, 10, 0.9);

        let x_train =
            Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let (forward, _) = dropout.forward(&x_train);
        assert_eq!(forward.shape(), x_train.shape());

        let x_train = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
        )
        .unwrap();
        let (forward1, _) = dropout.forward(&x_train);
        assert_eq!(forward1.shape(), x_train.shape());
        let (forward2, _) = dropout.forward(&x_train);
        assert_ne!(forward1, forward2);

        let mut dropout = DropOut::new(3, 10, 0.0);
        let x_train =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let (forward, _) = dropout.forward(&x_train);
        assert_eq!(forward, Array2::zeros((2, 3)));

        let mut dropout = DropOut::new(3, 10, 1.0);
        let x_train = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
        )
        .unwrap();
        let (forward, _) = dropout.forward(&x_train);
        assert_eq!(x_train, forward);
    }
}
