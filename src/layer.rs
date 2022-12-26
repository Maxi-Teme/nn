use dyn_clone::DynClone;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::functions::{relu, relu_grad, softmax, softmax_grad};

pub trait Layer: DynClone {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64>;

    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64>;

    fn predict(&self, inputs: Array2<f64>) -> Array2<f64>;
}

dyn_clone::clone_trait_object!(Layer);

/// dense network layer with weights and biases
#[derive(Debug, Clone)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,

    learning_rate: f64,

    last_inputs: Array2<f64>,
    dweights: Array2<f64>,
    dbiases: Array1<f64>,
    dinputs: Array2<f64>,
}

impl Dense {
    pub fn new(
        n_inputs: usize,
        size: usize,
        learning_rate: Option<f64>,
    ) -> Box<Self> {
        Box::new(Self {
            weights: Array2::<f64>::random(
                (n_inputs, size),
                Uniform::new(-1.0, 1.0),
            ),
            biases: Array1::<f64>::zeros(size),

            learning_rate: learning_rate.unwrap_or(0.1),

            last_inputs: Array2::<f64>::zeros((0, 0)),
            dweights: Array2::<f64>::zeros((0, 0)),
            dbiases: Array1::<f64>::zeros(0),
            dinputs: Array2::<f64>::zeros((0, 0)),
        })
    }
}

impl Layer for Dense {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        self.last_inputs = inputs;

        self.last_inputs.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        self.dweights = self.last_inputs.t().dot(dvalues);
        self.dbiases = dvalues.sum_axis(Axis(0));

        // compute dinputs before modifying weights
        self.dinputs = dvalues.dot(&self.weights.t());

        self.weights = &self.weights - self.learning_rate * &self.dweights;
        self.biases = &self.biases - self.learning_rate * &self.dbiases;

        self.dinputs.clone()
    }

    fn predict(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.dot(&self.weights) + &self.biases
    }
}

/// dropout network layer with dropout mask
#[derive(Debug, Clone)]
pub struct Dropout {
    mask: Array1<f64>,
    frac: (u32, u32),
}

impl Dropout {
    pub fn new(n_inputs: usize, numerator: u32, denominator: u32) -> Box<Self> {
        Box::new(Self {
            mask: Array1::<f64>::ones(n_inputs),
            frac: (numerator, denominator),
        })
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
}

impl Layer for Dropout {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        self.update_mask();

        inputs * &self.mask
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        dvalues * &self.mask
    }

    fn predict(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs * &self.mask
    }
}

/// rectified linear activation function layer
#[derive(Debug, Clone)]
pub struct ReLU {
    last_inputs: Array2<f64>,
}

impl ReLU {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            last_inputs: Array2::<f64>::zeros((0, 0)),
        })
    }
}

impl Layer for ReLU {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        self.last_inputs = inputs;

        relu(&self.last_inputs)
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        relu_grad(&self.last_inputs, dvalues)
    }

    fn predict(&self, inputs: Array2<f64>) -> Array2<f64> {
        relu(&inputs)
    }
}

/// Softmax activation function layer
#[derive(Debug, Clone)]
pub struct Softmax {
    last_inputs: Array2<f64>,
}

impl Softmax {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            last_inputs: Array2::<f64>::zeros((0, 0)),
        })
    }
}

impl Layer for Softmax {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        self.last_inputs = inputs.clone();

        softmax(&self.last_inputs)
    }

    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        softmax_grad(&self.last_inputs, dvalues)
    }

    fn predict(&self, inputs: Array2<f64>) -> Array2<f64> {
        softmax(&inputs)
    }
}
