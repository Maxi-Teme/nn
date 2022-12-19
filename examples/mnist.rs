use ndarray::{s, Array2, Axis};
use ndarray_rand::rand::{self, Rng};

use nn::{Activation, Layer, Model, Sequential};

pub fn load_mnist_dataset(
) -> ((Array2<f64>, Array2<usize>), (Array2<f64>, Array2<usize>)) {
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

    let y_train: Array2<usize> =
        Array2::from_shape_vec((y_train_raw.len(), 1), y_train_raw).unwrap();

    let x_test: Array2<f64> = Array2::from_shape_vec(
        (x_test_raw.len() / image_size, image_size),
        x_test_raw,
    )
    .unwrap();

    let y_test: Array2<usize> =
        ndarray::Array2::from_shape_vec((y_test_raw.len(), 1), y_test_raw)
            .unwrap();

    ((x_train, y_train), (x_test, y_test))
}

fn _pick_random_batch(
    x: &Array2<f64>,
    y: &Array2<f64>,
    batch_size: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut x_batch = Vec::with_capacity(batch_size);
    let mut y_batch = Vec::with_capacity(batch_size);
    let mut rng = rand::thread_rng();
    let mut taken_ids = Vec::with_capacity(batch_size);

    let x_size = x.shape()[0];

    if x_size <= batch_size {
        return (x.clone(), y.clone());
    }

    for _ in 0..batch_size.min(x_size) {
        let mut id: usize = 0;

        for _ in 0..100000 {
            id = rng.gen_range(0..x_size);
            if !taken_ids.contains(&id) {
                break;
            }
        }

        if !taken_ids.contains(&id) {
            x_batch.extend(x.slice(s![id, ..]).to_vec());
            y_batch.extend(y.slice(s![id, ..]).to_vec());
            taken_ids.push(id);
        }
    }

    (
        Array2::from_shape_vec((batch_size, x.shape()[1]), x_batch).unwrap(),
        Array2::from_shape_vec((batch_size, y.shape()[1]), y_batch).unwrap(),
    )
}

fn shuffle_dataset(x: &mut Array2<f64>, y: &mut Array2<f64>) {
    let examples = x.len_of(Axis(0));
    let input_size = x.len_of(Axis(1));

    let mut data_slice = x.as_slice_mut().unwrap();
    let mut label_slice = y.as_slice_mut().unwrap();

    for i in 0..examples - 1 {
        let new_index: usize = rand::thread_rng().gen_range(i..examples);

        let (data_indx_1, data_indx_2) =
            (i * input_size, new_index * input_size);
        // TODO Can we swap slices better?
        for t in 0..input_size {
            swap(&mut data_slice, data_indx_1 + t, data_indx_2 + t);
        }
        swap(&mut label_slice, i, new_index);
    }
}

fn swap<T: Copy>(list: &mut [T], a: usize, b: usize) {
    let temp = list[a];
    list[a] = list[b];
    list[b] = temp;
}

fn main() {
    let epochs = 5;
    let batch_size = 20000;

    let ((mut x_train, y_train_raw), (x_test, y_test)) = load_mnist_dataset();

    let mut y_train = Array2::zeros((y_train_raw.shape()[0], 10));
    for (row, y) in y_train_raw.axis_iter(Axis(0)).enumerate() {
        let y = y.get(0).unwrap();
        let one_hot_item = y_train.get_mut((row, *y)).unwrap();
        *one_hot_item = 1.0;
    }

    let mut model = Sequential::categorical();
    model.add_layer(Layer::dense(28 * 28, 128, Activation::relu()));
    model.add_layer(Layer::dense(128, 128, Activation::relu()));
    model.add_layer(Layer::dense(128, 10, Activation::softmax()));

    model.set_learning_rate(1.0);
    model.info(false);

    let x_size = x_train.shape()[0];
    println!("Training data has {} datapoints.\n", x_size);

    println!("Starting training\n");
    for e in 0..epochs {
        println!("Epoch: {}", e + 1);

        let mut losses = vec![];

        shuffle_dataset(&mut x_train, &mut y_train);

        for i in 0..(x_size / batch_size) {
            let start = i * batch_size;
            let end = start + batch_size;

            if end >= x_size - 1 {
                break;
            }

            let x_batch = x_train.slice(s![i..end, ..]).to_owned();
            let y_batch = y_train.slice(s![i..end, ..]).to_owned();

            let loss = model.train(&x_batch, &y_batch);
            println!("Batch {}-{} loss: {}", start, end, loss);
            losses.push(loss);
            model.info(false);
            model.peek_weight(0, (10, 10));
            model.peek_weight(1, (10, 10));
            model.peek_weight(2, (10, 10));
            println!("");
        }

        let epoch_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
        println!("Epoch loss: {}", epoch_loss);
        let accuracy = model.test(&x_test, &y_test);
        println!("Accuracy: {}\n", accuracy);
    }
}
