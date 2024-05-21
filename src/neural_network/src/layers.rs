pub mod relu {
    use ndarray::{Array, Ix2};

    #[derive(Debug, Clone)]
    pub struct ReluLayer {
        mask: Array<f64, Ix2>,
    }

    impl ReluLayer {
        pub fn new() -> ReluLayer {
            return ReluLayer { mask: Array::<f64, Ix2>::zeros((0, 0)) };
        }

        pub fn forward(&mut self, x: &Array<f64, Ix2>) -> Array<f64, Ix2> {
            self.mask = x.mapv(|x| if x <= 0.0 { 0.0 } else { 1.0 });
            let mut out = x * &self.mask;
            return out;
        }

        pub fn backward(&self, dout: &Array<f64, Ix2>) -> (Array<f64, Ix2>) {
            return dout * &self.mask;
        }
    }
}

pub mod sigmoid {
    use ndarray::{Array, Ix2};
    use crate::activation_functions::normal::sigmoid;

    #[derive(Debug, Clone)]
    pub struct SigmoidLayer {
        out: Array<f64, Ix2>,
    }

    impl SigmoidLayer {
        pub fn new() -> SigmoidLayer {
            return SigmoidLayer { out: Array::<f64, Ix2>::zeros((0, 0)) };
        }

        pub fn forward(&self, x: &Array<f64, Ix2>) -> Array<f64, Ix2> {
            let out = sigmoid(x);
            return out;
        }

        pub fn backward(&self, x: &Array<f64, Ix2>, dout: &Array<f64, Ix2>) -> (Array<f64, Ix2>) {
            let dx = dout  * (1.0 - &self.out) * &self.out;
            return dx;
        }
    }
}


pub mod affine {
    use ndarray::{Array, Axis, Ix1, Ix2};

    #[derive(Debug, Clone)]
    pub struct AffineLayer {
        w: Array<f64, Ix2>,
        b: Array<f64, Ix1>,
        x: Array<f64, Ix2>,
        pub(crate) dw: Array<f64, Ix2>,
        pub(crate) db: Array<f64, Ix1>,
    }

    impl AffineLayer {
        pub fn new(w: Array<f64, Ix2>, b: Array<f64, Ix1>) -> AffineLayer {
            return AffineLayer {
                w,
                b,
                x: Array::<f64, Ix2>::zeros((0, 0)),
                dw: Array::<f64, Ix2>::zeros((0, 0)),
                db: Array::<f64, Ix1>::zeros(0),
            };
        }

        pub fn forward(&mut self, x: &Array<f64, Ix2>) -> Array<f64, Ix2> {
            self.x = x.clone();
            let out = x.dot(&self.w) + &self.b;
            return out;
        }

        pub fn backward(&mut self, dout: &Array<f64, Ix2>) -> (Array<f64, Ix2>) {
            let dx = dout.dot(&self.w.t());
            self.dw = self.x.t().dot(dout);
            self.db = dout.sum_axis(Axis(0));
            return dx;
        }
    }
}

pub mod softmax_with_loss {
    use ndarray::{Array, Axis, Ix2, s};
    use crate::activation_functions::output::softmax;
    use crate::error_functions::normal::cross_entropy_error;

    #[derive(Debug, Clone)]
    pub struct SoftmaxWithLossLayer {
        loss: f64,
        y: Array<f64, Ix2>,
        t: Array<f64, Ix2>,
    }

    impl SoftmaxWithLossLayer {
        pub fn new() -> SoftmaxWithLossLayer {
            return SoftmaxWithLossLayer {
                loss: 0.0,
                y: Array::<f64, Ix2>::zeros((0, 0)),
                t: Array::<f64, Ix2>::zeros((0, 0)),
            };
        }

        pub fn forward(&mut self, x: &Array<f64, Ix2>, t: &Array<f64, Ix2>) -> f64 {
            self.t = t.clone();
            self.y = softmax(x);
            self.loss = cross_entropy_error(&self.y, &self.t);
            return self.loss;
        }

        pub fn backward(&self, dout: &f64) -> (Array<f64, Ix2>) {
            let batch_size = self.t.shape()[0] as f64;
            let dx = (&self.y - &self.t) / batch_size;
            return dx;
        }
    }
}
