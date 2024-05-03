pub mod normal {
    use ndarray::{Array, Array1, Array2, Dimension};
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

    // 一般のArrayに対するsigmoid関数
    pub fn sigmoid<D>(x: Array<f64, D>) -> Array<f64, D>
        where
            D: Dimension,
    {
        return x.map(|x| 1.0 / (1.0 + std::f64::consts::E.powf(-x)));
    }

    // 一般のArrayに対するReLU関数
    pub fn relu<D>(x: Array<f64, D>) -> Array<f64, D>
        where
            D: Dimension,{
        return x.map(|x| 0.0_f64.max(*x));
    }
}

pub mod output {
    use ndarray::{Array, Dimension};

    // 一般のArrayに対するidentity関数
    pub fn identity_function<D>(x: Array<f64, D>) -> Array<f64, D> {
        return x;
    }

    // 一般のArrayに対するsoftmax関数（overflow対策済み）
    pub fn softmax<D>(x: Array<f64, D>) -> Array<f64, D>
        where
            D: Dimension,{
        //xの要素の最大値を求める
        let max = x.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        //xの要素からmaxをそれぞれ引く
        let x = x.map(|x| x - max);
        let c = x.iter().fold(0.0, |acc, &x| acc + x.exp());
        let y = x.map(|x| x.exp() / c);
        return y;
    }
}