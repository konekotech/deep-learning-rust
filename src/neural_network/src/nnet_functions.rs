pub mod activation {
    use ndarray::{Array1, Array2};

    // Array1<f64> に対するstep関数
    pub fn step_function(x: Array1<f64>) -> Array1<f64> {
        let mut y = Array1::zeros(x.len());
        for i in 0..x.len() {
            if x[i] > 0.0 {
                y[i] = 1.0;
            } else {
                y[i] = 0.0;
            }
        }
        return y;
    }

    // Array1<f64> に対するsigmoid関数
    pub fn sigmoid(x: Array1<f64>) -> Array1<f64> {
        let mut y = Array1::zeros(x.len());
        for i in 0..x.len() {
            y[i] = 1.0 / (1.0 + std::f64::consts::E.powf(-x[i]));
        }
        return y;
    }

    // Array1<f64> に対するReLU関数
    pub fn relu(x: Array1<f64>) -> Array1<f64> {
        let mut y = Array1::zeros(x.len());
        for i in 0..x.len() {
            let a = 0.0_f64.max(x[i]);
            y[i] = a;
        }
        return y;
    }
}

pub mod output_activation {
    use ndarray::{Array1, Array2};

    // Array1<f64> に対するidentity関数
    pub fn identity_function(x: Array1<f64>) -> Array1<f64> {
        return x;
    }

    // Array1<f64> に対するsoftmax関数（overflow対策済み）
    pub fn softmax(x: Array1<f64>) -> Array1<f64> {
        //xの要素の最大値を求める
        let max = x.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        //xの要素からmaxをそれぞれ引く
        let x = x.map(|x| x - max);
        let c = x.iter().fold(0.0, |acc, &x| acc + x.exp());
        let y = x.map(|x| x.exp() / c);
        return y;
    }
}