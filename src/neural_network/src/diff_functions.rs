pub mod diff {
    use ndarray::{arr1, Array, Axis, Dimension, Ix1, Ix2};

    /// # Arguments
    /// * `f` - A function that takes a f64 and returns a f64
    /// * `x` - A f64
    pub fn numerical_diff<F>(f: F, x: f64) -> f64
        where F: Fn(f64) -> f64 {
        let h = 1e-4;
        return (f(x + h) - f(x - h)) / (2.0 * h);
    }

    /// # Arguments
    /// * `f` - A function that takes a 1D array of f64 and returns a f64
    /// * `x` - A 1D array of f64
    pub fn numerical_gradient1<F>(mut f: F, x: &Array<f64, Ix1>) -> Array<f64, Ix1>
        where F: FnMut(&Array<f64, Ix1>) -> f64 {
        let h = 1e-4;
        let mut x = x.to_owned();
        let mut grad = Array::zeros(x.dim().clone());
        for i in 0..x.len() {
            let tmp_val = x[i];
            x[i] = tmp_val + h;
            let fxh1 = f(&x);
            x[i] = tmp_val - h;
            let fxh2 = f(&x);
            grad[i] = (fxh1 - fxh2) / (2.0 * h);
            x[i] = tmp_val;
        }
        return grad;
    }

    /// # Arguments
    /// * `f` - A function that takes a 2D array of f64 and returns a f64
    /// * `x` - A 2D array of f64

    pub fn numerical_gradient2<F>(mut f: F, x: &Array<f64, Ix2>) -> Array<f64, Ix2>
        where F: FnMut(&Array<f64, Ix2>) -> f64 {
        let h = 1e-4;
        let mut x = x.to_owned();
        let mut grad = Array::zeros(x.dim().clone());
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let tmp_val = x[[i, j]];
                x[[i, j]] = tmp_val + h;
                let fxh1 = f(&x);
                x[[i, j]] = tmp_val - h;
                let fxh2 = f(&x);
                grad[[i, j]] = (fxh1 - fxh2) / (2.0 * h);
                x[[i, j]] = tmp_val;
            }
        }
        return grad;
    }


    /// # Arguments
    /// * `f` - A function that takes a 1D array of f64 and returns a f64
    /// * `init_x` - A 1D array of f64
    pub fn gradient_descent<F>(f: F, init_x: Array<f64, Ix1>, lr: f64, step_num: i64) -> Array<f64, Ix1>
        where F: Fn(&Array<f64, Ix1>) -> f64 {
        let mut x = init_x;
        for _ in 0..step_num {
            let grad = numerical_gradient1(&f, &x);
            x = x - lr * grad;
        }
        return x;
    }
}