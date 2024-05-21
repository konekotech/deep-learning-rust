pub mod two_layers {
    use ndarray::{Array, Array1, Axis, Dimension, Ix1, Ix2, s};
    use crate::activation_functions::normal::sigmoid;
    use crate::activation_functions::output::softmax;
    use crate::diff_functions::diff::{numerical_gradient1, numerical_gradient2};
    use crate::error_functions::normal::cross_entropy_error;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal};
    use crate::layers::affine::AffineLayer;
    use crate::layers::relu::ReluLayer;
    use crate::layers::softmax_with_loss::SoftmaxWithLossLayer;

    #[derive(Debug)]
    pub struct TwoLayerNet{
        pub w1: Array<f64, Ix2>,
        pub b1: Array<f64, Ix1>,
        pub w2: Array<f64, Ix2>,
        pub b2: Array<f64, Ix1>,
        layers: Layers,
        last_layer: SoftmaxWithLossLayer,
    }

    #[derive(Debug, Clone)]
    pub struct Layers {
        affine1: AffineLayer,
        relu: ReluLayer,
        affine2: AffineLayer,
    }

    impl TwoLayerNet{
        
        pub fn new( input_size: usize, hidden_size: usize, output_size: usize, weight_init_std: f64) -> TwoLayerNet
        {
            let w1 = weight_init_std * Array::<f64, Ix2>::random((input_size, hidden_size), Normal::new(0.0, 1.).unwrap());
            let b1 = Array::<f64, Ix1>::zeros(hidden_size);
            let w2 = weight_init_std * Array::<f64, Ix2>::random((hidden_size, output_size), Normal::new(0.0, 1.).unwrap());
            let b2 = Array::<f64, Ix1>::zeros(output_size);
            let affine1 = AffineLayer::new(w1.clone(), b1.clone());
            let relu = ReluLayer::new();
            let affine2 = AffineLayer::new(w2.clone(), b2.clone());
            let last_layer = SoftmaxWithLossLayer::new();
            let layers = Layers{affine1, relu, affine2};
            return TwoLayerNet{w1, b1, w2, b2, layers, last_layer};
        }

        pub fn predict(&mut self, x: &Array<f64, Ix2>) -> Array<f64, Ix2>{
            // 行ごとに処理してArray1に変換する
            // let mut row_arrays: Vec<Array1<_>> = Vec::new();
            // let mut y = Array::<f64, Ix2>::zeros((x.shape()[0], self.b2.len()));
            // for row in x.axis_iter(Axis(0)) {
            //     row_arrays.push(row.to_owned());
            // }
            // for (i, row) in row_arrays.iter().enumerate() {
                // let a1 = row.dot(&self.w1) + &self.b1;
                // let z1 = sigmoid(&a1);
                // let a2 = z1.dot(&self.w2) + &self.b2;
                // let y_row = softmax(&a2);
                // y_row = softmax(&a2);
                // y.slice_mut(s![i, ..]).assign(&y_row);
            //}
            let x = self.layers.affine1.forward(&x);
            let x = self.layers.relu.forward(&x);
            let x = self.layers.affine2.forward(&x);
            return x;
        }

        pub fn loss(&mut self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> f64{
            let y = self.predict(&x);
            return self.last_layer.forward(&y, &t);
        }

        pub fn accuracy(&mut self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> f64{
            let y = self.predict(x);
            // 最大値とそのインデックスを求める
            let mut accuracy = 0;
            for (i, y_row) in y.outer_iter().enumerate() {
                let (max_value, max_index) = y_row.iter().enumerate().fold((f64::NEG_INFINITY, 0), |acc, (index, &value)| {
                    if value > acc.0 {
                        (value, index)
                    } else {
                        acc
                    }
                });
                if(max_index == t[[i, 0]] as usize) {
                    accuracy += 1;
                }
            }
            return accuracy as f64 / x.shape()[0] as f64;
        }

        // pub fn numerical_gradient(&mut self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> TwoLayerNet{
        //     println!("{:?}", &self.w1.shape());
        //     let loss_w = |w: &Array<f64, Ix2>| self.loss(x, t);
        //     let loss_b = |b: &Array<f64, Ix1>| self.loss(x, t);
        //     let grad_w1 =  numerical_gradient2(loss_w, &self.w1);
        //     println!("{:?}", &grad_w1);
        //     let grad_b1 = numerical_gradient1(loss_b, &self.b1);
        //     let grad_w2 = numerical_gradient2(loss_w, &self.w2);
        //     let grad_b2 = numerical_gradient1(loss_b, &self.b2);
        //     return TwoLayerNet{w1: grad_w1, b1: grad_b1, w2: grad_w2, b2: grad_b2, layers: self.layers.clone(), last_layer: self.last_layer.clone()};
        // }

        pub fn gradient(&mut self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> TwoLayerNet{
            let _ = self.loss(&x, &t);
            let dout = 1.0;
            let dout = self.last_layer.backward(&dout);
            let dout = self.layers.affine2.backward(&dout);
            let dout = self.layers.relu.backward(&dout);
            let dout = self.layers.affine1.backward(&dout);
            return TwoLayerNet{w1: self.layers.affine1.dw.clone(), b1: self.layers.affine1.db.clone(), w2: self.layers.affine2.dw.clone(), b2: self.layers.affine2.db.clone(), layers: self.layers.clone(), last_layer: self.last_layer.clone()};
        }
    }
}