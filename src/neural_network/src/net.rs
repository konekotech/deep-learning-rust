pub mod two_layers {
    use ndarray::{Array, Array1, Axis, Dimension, Ix1, Ix2, s};
    use crate::activation_functions::normal::sigmoid;
    use crate::activation_functions::output::softmax;
    use crate::diff_functions::diff::{numerical_gradient1, numerical_gradient2};
    use crate::error_functions::normal::cross_entropy_error;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal};

    #[derive(Debug)]
    pub struct TwoLayerNet{
        pub w1: Array<f64, Ix2>,
        pub b1: Array<f64, Ix1>,
        pub w2: Array<f64, Ix2>,
        pub b2: Array<f64, Ix1>,
    }

    impl TwoLayerNet{
        
        pub fn new( input_size: usize, hidden_size: usize, output_size: usize, weight_init_std: f64) -> TwoLayerNet
        {
            let w1 = Array::<f64, Ix2>::random((input_size, hidden_size), Normal::new(0.0, weight_init_std).unwrap());
            let b1 = Array::<f64, Ix1>::zeros(hidden_size);
            let w2 = Array::<f64, Ix2>::random((hidden_size, output_size), Normal::new(0.0, weight_init_std).unwrap());
            let b2 = Array::<f64, Ix1>::zeros(output_size);
            return TwoLayerNet{w1, b1, w2, b2};
        }

        pub fn predict(&self, x: &Array<f64, Ix2>) -> Array<f64, Ix2>{
            // 行ごとに処理してArray1に変換する
            let mut row_arrays: Vec<Array1<_>> = Vec::new();
            let mut y = Array::<f64, Ix2>::zeros((x.shape()[0], self.b2.len()));
            for row in x.axis_iter(Axis(0)) {
                row_arrays.push(row.to_owned());
            }
            for (i, row) in row_arrays.iter().enumerate() {
                let a1 = row.dot(&self.w1) + &self.b1;
                let z1 = sigmoid(a1);
                let a2 = z1.dot(&self.w2) + &self.b2;
                let y_row = softmax(a2);
                y.slice_mut(s![i, ..]).assign(&y_row);
            }
            return y;
        }

        pub fn loss(&self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> f64{
            let y = self.predict(&x);
            let loss = cross_entropy_error(&y, &t);
            return loss;
        }

        pub fn accuracy(&self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> f64{
            let y = self.predict(&x);
            return 0.0;
        }

        pub fn numerical_gradient(&self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> TwoLayerNet{
            println!("{:?}", &self.w1.shape());
            let loss_w = |w: &Array<f64, Ix2>| self.loss(x, t);
            let loss_b = |b: &Array<f64, Ix1>| self.loss(x, t);
            let grad_w1 =  numerical_gradient2(loss_w, &self.w1);
            println!("{:?}", &grad_w1);
            let grad_b1 = numerical_gradient1(loss_b, &self.b1);
            let grad_w2 = numerical_gradient2(loss_w, &self.w2);
            let grad_b2 = numerical_gradient1(loss_b, &self.b2);
            return TwoLayerNet{w1: grad_w1, b1: grad_b1, w2: grad_w2, b2: grad_b2};
        }

    }



}