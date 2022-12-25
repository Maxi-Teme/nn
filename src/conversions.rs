use ndarray::{Array1, Array2, Axis};

pub fn one_hot_encode(ys: Array1<f64>) -> Array2<f64> {
    let max = ys.fold(0.0, |acc, y| y.max(acc)) as usize;
    let mut one_hot_encoded = Array2::<f64>::zeros((ys.len(), max + 1));

    for (idx, y) in ys.into_iter().enumerate() {
        if let Some(row) = one_hot_encoded.get_mut((idx, y as usize)) {
            *row = 1.0;
        } else {
            println!(
                "Could not find index ({}, {}) in shape {:?}",
                idx,
                y,
                one_hot_encoded.shape()
            );
        }
    }

    one_hot_encoded
}

pub fn one_hot_decode(ys: &Array2<f64>) -> Array1<f64> {
    ys.map_axis(Axis(1), |a| a.iter().position(|i| *i != 0.0).unwrap())
        .map(|a| *a as f64)
}
