use std::error::Error;
use std::fs::File;
use ndarray::{arr1, arr2};
use neural_network::activation_functions::normal::sigmoid;
use neural_network::activation_functions::output::softmax;
use mnist::*;
use ndarray::prelude::*;
use csv::{ReaderBuilder};
use ndarray_csv::Array2Reader;


#[test]
fn test_nnet() {
    // 3層ニューラルネットワークの実装
    let x = arr1(&[1.0, 0.5]);
    let w1 = arr2(&[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    let b1 = arr1(&[0.1, 0.2, 0.3]);
    println!("{:?}", x.shape());
    println!("{:?}", w1.shape());
    println!("{:?}", b1.shape());
    let a1 = x.dot(&w1) + &b1;
    let z1 = sigmoid(&a1);
    println!("{:?}", z1);
    let w2 = arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    let b2 = arr1(&[0.1, 0.2]);
    let a2 = z1.dot(&w2) + &b2;
    let z2 = sigmoid(&a2);
    println!("{:?}", z2);
    let w3 = arr2(&[[0.1, 0.3], [0.2, 0.4]]);
    let b3 = arr1(&[0.1, 0.2]);
    let a3 = z2.dot(&w3) + &b3;
    let y = a3;
    println!("{:?}", y);
    let a = arr1(&[1010.0, 1000.0, 990.0]);
    println!("{:?}", softmax(&a));
}

fn read_mnist() -> (Array<f64, Ix3>, Array2<f64>, Array<f64, Ix3>, Array2<f64>) {
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
    (train_data, train_labels, _test_data, _test_labels)
}

fn read_csv_to_array1(file_name: String) -> Result<Array1<f64>, Box<dyn Error>> {
    let array = read_csv_to_array2(file_name).unwrap();
    Ok(array.into_raw_vec().into())
}

fn read_csv_to_array2(file_name: String) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(File::open(file_name).unwrap());
    let array: Array2<f64> = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

/// Returns the weights and biases for the neural network
/// # Returns
/// (w1, w2, w3, b1, b2, b3)
fn read_weights() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>){
    let file_name = "./data/sample_weight/W1.csv";
    let w1 = read_csv_to_array2(file_name.to_string()).unwrap();
    let file_name = "./data/sample_weight/W2.csv";
    let w2 = read_csv_to_array2(file_name.to_string()).unwrap();
    let file_name = "./data/sample_weight/W3.csv";
    let w3 = read_csv_to_array2(file_name.to_string()).unwrap();
    let file_name = "./data/sample_weight/b1.csv";
    let b1 = read_csv_to_array1(file_name.to_string()).unwrap();
    let file_name = "./data/sample_weight/b2.csv";
    let b2 = read_csv_to_array1(file_name.to_string()).unwrap();
    let file_name = "./data/sample_weight/b3.csv";
    let b3 = read_csv_to_array1(file_name.to_string()).unwrap();
    (w1, w2, w3, b1, b2, b3)
}


#[test]
fn test_mnist_learning() {
    let (w1, w2, w3, b1, b2, b3) = read_weights();
    let (x_train, t_train, x_test, t_test) = read_mnist();
    //x_testは3次元だが、2次元に変換する。[10000, 784]に変換する
    let x_test = x_test.into_shape((10000, 784)).unwrap();

    // 行ごとに処理してArray1に変換する
    let mut row_arrays: Vec<Array1<_>> = Vec::new();
    for row in x_test.axis_iter(Axis(0)) {
        row_arrays.push(row.to_owned());
    }
    let mut accuracy = 0;
    for (i, row) in row_arrays.iter().enumerate() {
        let a1 = row.dot(&w1) + &b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&w2) + &b2;
        let z2 = sigmoid(&a2);
        let a3 = z2.dot(&w3) + &b3;
        let y = softmax(&a3);
        // 最大値とそのインデックスを求める
        let (max_value, max_index) = y.iter().enumerate().fold((f64::NEG_INFINITY, 0), |acc, (index, &value)| {
            if value > acc.0 {
                (value, index)
            } else {
                acc
            }
        });
        if(max_index == t_test[[i, 0]] as usize) {
            accuracy += 1;
        }
    }
    println!("accuracy is {:?}", accuracy as f64 / 10000.0);
}

#[test]
fn test_learning_with_batch(){
    let (w1, w2, w3, b1, b2, b3) = read_weights();
    let (x_train, t_train, x_test, t_test) = read_mnist();
    //x_testは3次元だが、2次元に変換する。[10000, 784]に変換する
    let x_test = x_test.into_shape((10000, 784)).unwrap();
    let mut accuracy = 0;
    let batch_size = 1000;
    for i in 0..10000/batch_size {
        let x_batch = x_test.slice(s![i*batch_size..(i+1)*batch_size, ..]);
        let t_batch = t_test.slice(s![i*batch_size..(i+1)*batch_size, ..]);
        let a1 = x_batch.dot(&w1) + &b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&w2) + &b2;
        let z2 = sigmoid(&a2);
        let a3 = z2.dot(&w3) + &b3;
        let y = softmax(&a3);
        // 最大値とそのインデックスを求める
        for (j, y_row) in y.outer_iter().enumerate() {
            let (max_value, max_index) = y_row.iter().enumerate().fold((f64::NEG_INFINITY, 0), |acc, (index, &value)| {
                if value > acc.0 {
                    (value, index)
                } else {
                    acc
                }
            });
            if(max_index == t_batch[[j, 0]] as usize) {
                accuracy += 1;
            }
        }
    }
    println!("accuracy is {:?}", accuracy as f64 / 10000.0);
}



