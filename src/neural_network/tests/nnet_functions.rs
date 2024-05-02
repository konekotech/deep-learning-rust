use std::error::Error;
use std::fs::File;
use ndarray::{arr1, arr2, OwnedRepr};
use neural_network::nnet_functions::activation::sigmoid;
use neural_network::nnet_functions::output_activation::softmax;
use mnist::*;
use ndarray::prelude::*;
use csv::{ReaderBuilder};
use ndarray_csv::Array2Reader;


#[test]
fn test_mnist() {
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
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);
}

fn read_csv_to_array2(file_name: String) -> Result<Array2<f64>, Box<dyn Error>> {
    //Array2に変換する
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(File::open(file_name).unwrap());
    let array: Array2<f64> = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn read_csv_to_array1(file_name: String) -> Result<Array1<f64>, Box<dyn Error>> {
    let array = read_csv_to_array2(file_name).unwrap();
    Ok(array.into_raw_vec().into())
}

#[test]
fn test_read_csv(){
    let file_name = "./data/sample_weight/W1.csv";
    let w1 = read_csv_to_array2(file_name.to_string()).unwrap();
    println!("{:?}", w1.shape());
    let file_name = "./data/sample_weight/W2.csv";
    let w2 = read_csv_to_array2(file_name.to_string()).unwrap();
    println!("{:?}", w2.shape());
    let file_name = "./data/sample_weight/b1.csv";
    let b1 = read_csv_to_array1(file_name.to_string()).unwrap();
    println!("{:?}", b1.shape());
}






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
    let z1 = sigmoid(a1);
    println!("{:?}", z1);
    let w2 = arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    let b2 = arr1(&[0.1, 0.2]);
    let a2 = z1.dot(&w2) + &b2;
    let z2 = sigmoid(a2);
    println!("{:?}", z2);
    let w3 = arr2(&[[0.1, 0.3], [0.2, 0.4]]);
    let b3 = arr1(&[0.1, 0.2]);
    let a3 = z2.dot(&w3) + &b3;
    let y = a3;
    println!("{:?}", y);
    let a = arr1(&[1010.0, 1000.0, 990.0]);
    println!("{:?}", softmax(a));
}


