extern crate blas_src;

use nn::{Layer, Loss, Model, Sequential};

use nn::conversions::one_hot_encode;
use nn::helpers;

fn main() {
    let n_epochs = 10000;
    let learning_rate = 1.0;

    let n_inputs = 2;
    let hidden_layer_size = 64;
    let n_outputs = 3;

    let mut model = Sequential::new();
    model.add_layer(Layer::new_dense(
        n_inputs,
        hidden_layer_size,
        Some(learning_rate),
    ));
    model.add_layer(Layer::new_relu());
    model.add_layer(Layer::new_dense(
        hidden_layer_size,
        hidden_layer_size,
        Some(learning_rate),
    ));
    model.add_layer(Layer::new_relu());
    model.add_layer(Layer::new_dense(
        hidden_layer_size,
        n_outputs,
        Some(learning_rate),
    ));
    model.set_loss_fn(Loss::new_scce());

    let (x, y) = helpers::spiral_data(100, 3);

    let y = one_hot_encode(y);

    for e in 1..n_epochs + 1 {
        model.fit(&x, &y);

        if e % 200 == 0 {
            println!("\nepoch {}/{}", e, n_epochs);
            println!("loss: {}", &model.loss(&x, &y));
            println!("acc: {}", &model.accuracy(&x, &y));
        }
    }
}
