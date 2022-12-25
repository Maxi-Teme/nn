use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use nn::conversions::one_hot_encode;
use nn::functions::{
    accuracy, ccr_mean, relu, relu_grad, softmax, softmax_and_ccr_grad,
};
use nn::helpers;

fn main() {
    let n_epochs = 10000;
    let learning_rate = 1.0;

    let n_inputs = 2;
    let hidden_layer_size = 64;
    let n_outputs = 3;

    let mut weights_1 = Array2::<f64>::random(
        (n_inputs, hidden_layer_size),
        Uniform::new(-0.1, 0.1),
    );
    let mut biases_1 = Array2::<f64>::zeros((1, hidden_layer_size));

    let mut weights_2 = Array2::<f64>::random(
        (hidden_layer_size, hidden_layer_size),
        Uniform::new(-0.1, 0.1),
    );
    let mut biases_2 = Array2::<f64>::zeros((1, hidden_layer_size));

    let mut weights_out = Array2::<f64>::random(
        (hidden_layer_size, n_outputs),
        Uniform::new(-0.1, 0.1),
    );
    let mut biases_out = Array2::<f64>::zeros((1, n_outputs));

    let (x, y) = helpers::spiral_data(100, 3);
    let y = one_hot_encode(y);

    for e in 1..n_epochs + 1 {
        let z_1 = x.dot(&weights_1) + &biases_1;
        let a_1 = relu(&z_1);

        let z_2 = a_1.dot(&weights_2) + &biases_2;
        let a_2 = relu(&z_2);

        let z_out = a_2.dot(&weights_out) + &biases_out;
        // let z_out = a_1.dot(&weights_out) + &biases_out;
        let a_out = softmax(&z_out);

        let loss = ccr_mean(&a_out, &y);
        let accuracy = accuracy(&a_out, &y);

        if e % 200 == 0 {
            println!("\nepoch {}/{}", e, n_epochs);
            println!("loss: {}", &loss);
            println!("acc: {}", &accuracy);
        }

        let grad_out = softmax_and_ccr_grad(&z_out, &y);

        let d_weights_out = a_2.t().dot(&grad_out);
        // let d_weights_out = a_1.t().dot(&grad_out);
        let d_biases_out = grad_out.sum_axis(Axis(0)).insert_axis(Axis(0));
        let d_inputs_out = &grad_out.dot(&weights_out.t());
        weights_out = weights_out - learning_rate * d_weights_out;
        biases_out = biases_out - learning_rate * d_biases_out;

        let grad_2 = relu_grad(&z_2, d_inputs_out);

        let d_weights_2 = a_1.t().dot(&grad_2);
        let d_biases_2 = grad_2.sum_axis(Axis(0)).insert_axis(Axis(0));
        let d_inputs_2 = &grad_2.dot(&weights_2.t());
        weights_2 = weights_2 - learning_rate * d_weights_2;
        biases_2 = biases_2 - learning_rate * d_biases_2;

        let grad_1 = relu_grad(&z_1, d_inputs_2);
        // let grad_1 = relu_grad(&z_1, d_inputs_out);

        let d_weights_1 = x.t().dot(&grad_1);
        let d_biases_1 = grad_1.sum_axis(Axis(0)).insert_axis(Axis(0));
        weights_1 = weights_1 - learning_rate * d_weights_1;
        biases_1 = biases_1 - learning_rate * d_biases_1;
    }
}
