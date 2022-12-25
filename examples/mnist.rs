use ndarray::{s, Array2};
use nn::{Activation, Layer, Model, Sequential};

pub fn load_mnist_dataset(
) -> ((Array2<f64>, Array2<f64>), (Array2<f64>, Array2<f64>)) {
    let image_size = 28 * 28;

    let x_train_raw: Vec<f64> =
        mnist_read::read_data("data/mnist/train-images.idx3-ubyte")
            .into_iter()
            .map(|d| d as f64 / 255.0)
            .collect();

    let y_train_raw: Vec<usize> =
        mnist_read::read_labels("data/mnist/train-labels.idx1-ubyte")
            .into_iter()
            .map(|l| l as usize)
            .collect();

    let x_test_raw: Vec<f64> =
        mnist_read::read_data("data/mnist/t10k-images.idx3-ubyte")
            .into_iter()
            .map(|d| d as f64 / 255.0)
            .collect();

    let y_test_raw: Vec<usize> =
        mnist_read::read_labels("data/mnist/t10k-labels.idx1-ubyte")
            .into_iter()
            .map(|l| l as usize)
            .collect();

    let x_train: Array2<f64> = Array2::from_shape_vec(
        (x_train_raw.len() / image_size, image_size),
        x_train_raw,
    )
    .unwrap();

    let y_train_raw: Vec<f64> = y_train_raw
        .into_iter()
        .map(|y| {
            let mut one_hot = Vec::with_capacity(10);
            for _ in 0..10 {
                one_hot.push(0.0);
            }
            one_hot[y as usize] = 1.0;
            one_hot
        })
        .flatten()
        .collect();
    let y_train: Array2<f64> =
        Array2::from_shape_vec((y_train_raw.len() / 10, 10), y_train_raw)
            .unwrap();

    println!(
        "\nx_train: {:?}, y_train: {:?}\n",
        x_train.shape(),
        y_train.shape()
    );

    let x_test: Array2<f64> = Array2::from_shape_vec(
        (x_test_raw.len() / image_size, image_size),
        x_test_raw,
    )
    .unwrap();
    let y_test_raw: Vec<f64> = y_test_raw
        .into_iter()
        .map(|y| {
            let mut one_hot = Vec::with_capacity(10);
            for _ in 0..10 {
                one_hot.push(0.0);
            }
            one_hot[y as usize] = 1.0;
            one_hot
        })
        .flatten()
        .collect();
    let y_test: Array2<f64> =
        Array2::from_shape_vec((y_test_raw.len() / 10, 10), y_test_raw)
            .unwrap();

    println!(
        "\nx_test: {:?}, y_test: {:?}\n",
        x_test.shape(),
        y_test.shape()
    );

    ((x_train, y_train), (x_test, y_test))
}

fn main() {
    let epochs = 10;
    let batch_size = 10000;

    let ((x_train, y_train), _) = load_mnist_dataset();

    // let x_train = x_train.slice(s![0..20000, ..]).to_owned();
    // let y_train = y_train.slice(s![0..20000, ..]).to_owned();

    let mut model = Sequential::categorical();
    model.add_layer(Layer::dense(28 * 28, 200, Activation::relu()));
    // model.add_layer(Layer::dense(200, 200, Activation::relu()));
    model.add_layer(Layer::dense(200, 10, Activation::softmax()));
    model.set_learning_rate(0.1);

    println!("Training data has {} datapoints.\n", x_train.shape()[0]);
    println!("Starting training\n");

    let number_of_batches =
        (x_train.shape()[0] as f64 / batch_size as f64).floor() as i32;

    println!(
        "\nWill train {} epochs with {} batches of size {}\n",
        epochs, number_of_batches, batch_size
    );

    for e in 0..epochs {
        println!("Epoch: {}", e + 1);

        let mut losses = vec![];

        for i in 0..number_of_batches {
            let start = i * batch_size;
            let end = start + batch_size;

            let x_batch = x_train.slice(s![start..end, ..]).to_owned();
            let y_batch = y_train.slice(s![start..end, ..]).to_owned();

            println!(
                "x_batch: {:?}, y_batch: {:?}: {}/{}",
                x_batch.shape(),
                y_batch.shape(),
                i + 1,
                number_of_batches,
            );

            let loss = model.train(&x_batch, &y_batch);

            losses.push(loss);
        }

        // let accuracy = model.test(&x_test, &y_test);

        println!("Epoch losses: {:?}", losses);
        // println!("Accuracy: {}\n", accuracy);
    }
}
