use ndarray::{Array1, Array2, Axis};
use ndarray_stats::QuantileExt;

pub fn relu(inputs: &Array2<f64>) -> Array2<f64> {
    inputs.map(|i| i.max(0.0))
}

pub fn relu_grad(finputs: &Array2<f64>, dvalues: &Array2<f64>) -> Array2<f64> {
    dvalues * finputs.map(|i| if *i <= 0.0 { 0.0 } else { 1.0 })
}

/// from: https://aimatters.wordpress.com/2020/06/14/derivative-of-softmax-layer/
pub fn softmax(inputs: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros(inputs.raw_dim());

    for (mut result, row) in
        result.axis_iter_mut(Axis(0)).zip(inputs.axis_iter(Axis(0)))
    {
        let max = row.fold(f64::EPSILON, |acc, x| x.max(acc));
        let exps = row.map(|x| (x - max).exp());

        result.assign(&(&exps / exps.sum()));
    }

    result
}

/// foutputs: by softmax activated values
pub fn softmax_grad(
    foutputs: &Array2<f64>,
    dvalues: &Array2<f64>,
) -> Array2<f64> {
    let mut dinputs = Array2::<f64>::zeros(dvalues.raw_dim());

    for (idx, (single_output, single_dvalues)) in foutputs
        .axis_iter(Axis(0))
        .zip(dvalues.axis_iter(Axis(0)))
        .enumerate()
    {
        let single_output_m = single_output.insert_axis(Axis(1));
        let jacobian_matrix = Array2::from_diag(&single_output)
            - single_output_m.dot(&single_output_m.t());

        dinputs
            .row_mut(idx)
            .assign(&jacobian_matrix.dot(&single_dvalues));
    }

    dinputs
}

pub fn softmax_and_ccr_grad(
    dvalues: &Array2<f64>,
    targets: &Array2<f64>,
) -> Array2<f64> {
    let mut dinputs = dvalues.clone();

    let samples = dvalues.nrows();

    let y_true = targets
        .map_axis(Axis(1), |a| a.iter().position(|i| *i != 0.0).unwrap());

    for (mut drow, y) in dinputs.axis_iter_mut(Axis(0)).zip(y_true.iter()) {
        let update = drow.get_mut(*y).unwrap();
        *update -= 1.0;
    }

    dinputs / samples as f64
}

// cost functions
//

pub fn ccr_mean(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    ccr(predictions, targets).mean().unwrap()
}

fn ccr(predictions: &Array2<f64>, targets: &Array2<f64>) -> Array1<f64> {
    -(predictions * targets)
        .sum_axis(Axis(1))
        .map(|i| i.clamp(f64::EPSILON, 1.0 - f64::EPSILON).ln())
}

pub fn ccr_grad(dvalues: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
    let dinputs = -targets / dvalues;
    dinputs / targets.shape()[0] as f64
}

// accuracy
//

pub fn accuracy(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let correct_classes =
        argmax(predictions).map(|i| *i as f64).insert_axis(Axis(1));

    (correct_classes * targets).mean().unwrap()
}

fn argmax(predictions: &Array2<f64>) -> Array1<usize> {
    predictions.map_axis(Axis(1), |a| a.argmax_skipnan().unwrap())
}

#[cfg(test)]
mod test {
    use ndarray::{Array2, Axis};

    use super::{ccr_grad, softmax_and_ccr_grad, softmax_grad};

    #[test]
    fn derive_ccr_then_softmax() {
        let softmax_outputs = Array2::<f64>::from_shape_vec(
            (3, 3),
            vec![0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08],
        )
        .unwrap();

        let class_targets = Array2::<f64>::from_shape_vec(
            (3, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();

        let dvalues1 = softmax_and_ccr_grad(&softmax_outputs, &class_targets);

        let dinputs2 = ccr_grad(&softmax_outputs, &class_targets);
        let dvalues2 = softmax_grad(&softmax_outputs, &dinputs2);

        println!("{}", &dvalues1);
        println!("{}", &dvalues2);

        for (one, two) in dvalues1.into_iter().zip(dvalues2.into_iter()) {
            assert!((one - two).abs() < 10e-16, "|{} - {}| > 10e-16", one, two);
        }
    }

    #[test]
    fn one_hot_to_spartial() {
        let one_hot = Array2::<f64>::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();

        let spartial = one_hot
            .map_axis(Axis(1), |a| a.iter().position(|i| *i != 0.0).unwrap());

        dbg!(&spartial);
    }
}
