use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use nn::functions::{
    accuracy, ccr_mean, relu, relu_grad, softmax, softmax_and_ccr_grad,
};

fn main() {
    let n_epochs = 1000;
    let learning_rate = 0.2;

    let n_inputs = 2;
    let hidden_layer_size = 400;
    let n_outputs = 3;

    let mut weights_1 = Array2::<f64>::random(
        (n_inputs, hidden_layer_size),
        Uniform::new(-0.1, 0.1),
    );
    let mut biases_1 = Array2::<f64>::zeros((1, hidden_layer_size));

    let mut weights_out = Array2::<f64>::random(
        (hidden_layer_size, n_outputs),
        Uniform::new(-0.1, 0.1),
    );
    let mut biases_out = Array2::<f64>::zeros((1, n_outputs));

    let (x, y) = spiral_data(100, 3);
    let y = one_hot_encode(y);

    for e in 1..n_epochs + 1 {
        let z_1 = x.dot(&weights_1) + &biases_1;
        let a_1 = relu(&z_1);

        let z_out = a_1.dot(&weights_out) + &biases_out;
        let a_out = softmax(&z_out);

        let loss = ccr_mean(&a_out, &y);

        let accuracy = accuracy(&a_out, &y);
        if e % 250 == 0 {
            println!("\nepoch {}/{}", e, n_epochs);
            println!("loss: {}", &loss);
            println!("acc: {}", &accuracy);
        }

        let grad_out = softmax_and_ccr_grad(&z_out, &y);

        let d_weights_out = a_1.t().dot(&grad_out);
        let d_biases_out = grad_out.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_1 = relu_grad(&z_1, &grad_out.dot(&weights_out.t()));

        let d_weights_1 = x.t().dot(&grad_1);
        let d_biases_1 = grad_1.sum_axis(Axis(0)).insert_axis(Axis(0));

        weights_out = weights_out - learning_rate * d_weights_out;
        biases_out = biases_out - learning_rate * d_biases_out;

        weights_1 = weights_1 - learning_rate * d_weights_1;
        biases_1 = biases_1 - learning_rate * d_biases_1;
    }
}

pub fn spiral_data(
    points: usize,
    classes: usize,
) -> (Array2<f64>, Array1<f64>) {
    let mut y: Array<f64, ndarray::Dim<[usize; 1]>> =
        Array::zeros(points * classes);
    let mut x = Vec::with_capacity(points * classes * 2);

    for class_number in 0..classes {
        let r = Array::linspace(0.0, 1.0, points);
        let t = (Array::linspace(
            (class_number * 4) as f64,
            ((class_number + 1) * 4) as f64,
            points,
        ) + Array::random(points, Normal::new(0.0, 1.0).unwrap())
            * 0.2)
            * 2.5;
        let r2 = r.clone();
        let mut c = Vec::<f64>::new();
        for (x, y) in (r * t.map(|x| (x).sin()))
            .into_raw_vec()
            .iter()
            .zip((r2 * t.map(|x| (x).cos())).into_raw_vec().iter())
        {
            c.push(*x);
            c.push(*y);
        }
        for (ix, n) in ((points * class_number)..(points * (class_number + 1)))
            .zip((0..).step_by(2))
        {
            x.push(c[n]);
            x.push(c[n + 1]);
            y[ix] = class_number as f64;
        }
    }
    (
        ndarray::ArrayBase::from_shape_vec((points * classes, 2), x).unwrap(),
        y,
    )
}

fn one_hot_encode(ys: Array1<f64>) -> Array2<f64> {
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

fn one_hot_decode(ys: Array2<f64>) -> Array1<f64> {
    ys.map_axis(Axis(1), |a| a.iter().position(|i| *i != 0.0).unwrap())
        .map(|a| *a as f64)
}
