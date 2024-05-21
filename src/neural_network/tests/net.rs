use ndarray::{Array, Ix1, Ix2, s};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use neural_network::net::two_layers::TwoLayerNet;
use neural_network::mnist::load_mnist;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SliceRandom;
use ndarray_rand::rand::thread_rng;

#[test]
fn test_nnet() {
    let mut net = TwoLayerNet::new(784, 100, 10, 0.01);
    //shapeの確認
    assert_eq!(net.w1.shape(), &[784, 100]);
    assert_eq!(net.b1.shape(), &[100]);
    assert_eq!(net.w2.shape(), &[100, 10]);
    assert_eq!(net.b2.shape(), &[10]);

    let x =  Array::<f64, Ix2>::random((100, 784), Uniform::new(0., 1.));
    let y = TwoLayerNet::predict(&mut net, &x);
    println!("{:?}", y);
    let t =  Array::<f64, Ix2>::random((100,10), Uniform::new(0., 1.));
}

# [test]
fn test_mini_batch_with_numerical_gradient(){
    // 死ぬほど遅い

    let (x_train, t_train, x_test, t_test) = load_mnist();
    let itrs_num = 100;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    for i in 0..itrs_num {
        // ランダムなインデックスを生成
        let mut rng = thread_rng();
        let batch_mask: Vec<usize> = (0..train_size).collect();
        let batch_mask = batch_mask.choose_multiple(&mut rng, batch_size).cloned().collect::<Vec<usize>>();

        // ミニバッチを作成
        let mut x_batch = Array2::<f64>::zeros((batch_size, x_train.shape()[1]));
        let mut t_batch = Array2::<f64>::zeros((batch_size, t_train.shape()[1]));

        for (i, &j) in batch_mask.iter().enumerate() {
            x_batch.slice_mut(s![i, ..]).assign(&x_train.slice(s![j, ..]));
            t_batch.slice_mut(s![i, ..]).assign(&t_train.slice(s![j, ..]));
        }

        //let grad = network.numerical_gradient(&x_batch, &t_batch);
        let grad = network.gradient(&x_batch, &t_batch);
        println!("{:?}", grad.b1);
        network.w1 = network.w1 - learning_rate * grad.w1;
        network.b1 = network.b1 - learning_rate * grad.b1;
        network.w2 = network.w2 - learning_rate * grad.w2;
        network.b2 = network.b2 - learning_rate * grad.b2;
        let loss = network.loss(&x_batch, &t_batch);
        println!("loss is {:?}", loss);
    }
}