use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

mod constants;

pub use constants::*;
use ndarray_stats::QuantileExt;

pub fn get_random_data_array(shape: (usize, usize)) -> Array2<f64> {
    Array2::random(shape, Uniform::new(0.0, 1.0))
}

pub fn get_random_one_hot_array(shape: (usize, usize)) -> Array2<f64> {
    let mut arr = Array2::random(shape, Uniform::new(0.0, 1.0));
    arr.map_axis_mut(Axis(1), |mut i| {
        let max = i.max().unwrap().clone();
        i.map_inplace(|j| if *j == max { *j = 1.0 } else { *j = 0.0 })
    });

    arr
}
