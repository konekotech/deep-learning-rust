pub mod perceptron {
    pub fn add(x1: f64, x2: f64) -> f64 {
        let w = [0.5, 0.5];
        let b = -0.7;
        let tmp = x1 * w[0] + x2 * w[1] + b;
        if tmp <= 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
    }
    fn nand(x1: f64, x2: f64) -> f64 {
        let w = [-0.5, -0.5];
        let b = 0.7;
        let tmp = x1 * w[0] + x2 * w[1] + b;
        if tmp <= 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    // Vec<f64> に対するstep関数
    fn step_function(x: Vec<f64>) -> Vec<f64> {
        let mut y = Vec::new();
        for i in x {
            if i > 0.0 {
                y.push(1.0);
            } else {
                y.push(0.0);
            }
        }
        return y;
    }

    // Vec<f64> に対するsigmoid関数
    fn sigmoid(x: Vec<f64>) -> Vec<f64> {
        let mut y = Vec::new();
        for i in x {
            y.push(1.0 / (1.0 + std::f64::consts::E.powf(-i)));
        }
        return y;
    }

    // Vec<f64> に対するReLU関数
    fn relu(x: Vec<f64>) -> Vec<f64> {
        let mut y = Vec::new();
        for i in x {
            let a = 0.0_f64.max(i);
            y.push(a);
        }
        return y;
    }
}
