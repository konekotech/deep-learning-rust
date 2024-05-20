use ndarray::{arr1, Array, Ix1, Ix2};
use neural_network::diff_functions::diff::{gradient_descent, numerical_diff, numerical_gradient1};


#[test]
fn test_numerical_diff() {
    let f = |x: f64| x.powi(2);
    let x = 3.0;
    let result = numerical_diff(f, x);
    //誤差が0.00001以内であればOKにする
    let result = numerical_diff(f, 3.0);
    assert!((result - 6.0).abs() < 0.00001);
}

#[test]
fn test_numerical_gradient() {
    let f = |x: &Array<f64, Ix1>| x[0].powi(2) + x[1].powi(2);
    let x = arr1(&[3.0, 4.0]);
    let result = numerical_gradient1(f, &x);
    let expected = arr1(&[6.0, 8.0]);
    //誤差が0.00001以内であればOKにする
    assert!((result - expected).iter().all(|&x| x.abs() < 0.00001));
}

#[test]
fn test_gradient_descent (){
    let f = |x: &Array<f64, Ix1>| x[0].powi(2) + x[1].powi(2);
    let init_x = arr1(&[-3.0, 4.0]);
    let lr = 10.0;
    let step_num = 100;
    let result = gradient_descent(f, init_x, lr, step_num);
    let expected = arr1(&[-2.58983747e+13, -1.29524862e+12]);
    //誤差が0.00001以内であればOKにする
    println!("{:?}", result);
    assert!((result - expected).iter().all(|&x| x.abs() < 100000.0));
}

