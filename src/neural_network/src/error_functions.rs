pub mod normal{
    use ndarray::{Array, Dimension};

    /// # Arguments
    /// * `y` - A 1D array of f64
    /// * `t` - A 1D array of f64
    pub fn mean_squared_error<D>(y: Array<f64, D>, t: Array<f64, D>) -> f64
    where D: Dimension{
        let diff = &y - &t;
        let sum = diff.iter().fold(0.0, |acc, &x| acc + x.powi(2));
        return sum / 2.0;
    }

    /// # Arguments
    /// * `y` - A 1D array of f64
    /// * `t` - A 1D array of f64
    pub fn cross_entropy_error<D>(y: Array<f64, D>, t: Array<f64, D>) -> f64
    where D: Dimension{
        let delta = 1e-7;
        let y = y.map(|x| x + delta);
        let sum = y.iter().zip(t.iter()).fold(0.0, |acc, (&y, &t)| acc - t * y.ln());
        return sum;
    }
}