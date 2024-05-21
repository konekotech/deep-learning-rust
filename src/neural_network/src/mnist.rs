use mnist::{Mnist, MnistBuilder};
use ndarray::{Array, Array2, Array3, Axis, Ix2};

/// Converts a vector of labels to one-hot encoding
fn one_hot_encode(labels: Array2<f64>, num_classes: usize) -> Array2<f64> {
    let num_labels = labels.len();
    let mut one_hot = Array2::<f64>::zeros((num_labels, num_classes));

    for (i, label) in labels.iter().enumerate() {
        let class = *label as usize;
        one_hot[[i, class]] = 1.0;
    }

    one_hot
}

pub fn load_mnist() -> (Array<f64, Ix2>, Array2<f64>, Array<f64, Ix2>, Array2<f64>) {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Convert images to Array3 format and normalize
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    // Convert labels to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    // Reshape data for use in neural networks
    let train_data = train_data.into_shape((50_000, 784)).unwrap();
    let test_data = test_data.into_shape((10_000, 784)).unwrap();

    // One-hot encode the labels
    let train_labels_one_hot = one_hot_encode(train_labels, 10);
    let test_labels_one_hot = one_hot_encode(test_labels, 10);

    (train_data, train_labels_one_hot, test_data, test_labels_one_hot)
}


pub fn load_mnist_without_onehot() -> (Array<f64, Ix2>, Array2<f64>, Array<f64, Ix2>, Array2<f64>) {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);
    // println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));
    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);
    // println!("The first digit is a {:?}", train_labels.slice(s![image_num, ..]));
    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);
    let _test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);
    //4つのArrayを返す
    let train_data = train_data.into_shape((50000, 784)).unwrap();
    let _test_data = _test_data.into_shape((10000, 784)).unwrap();
    (train_data, train_labels, _test_data, _test_labels)
}