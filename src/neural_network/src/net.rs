pub mod two_layers {
    use ndarray::{Array, Dimension};

    struct TwoLayerNet<D> where D: Dimension{
        pub w1: Array<f64, D>,
        pub b1: Array<f64, D>,
        pub w2: Array<f64, D>,
        pub b2: Array<f64, D>,
    }

    fn new<D>(w1: Array<f64, D>, b1: Array<f64, D>, w2: Array<f64, D>, b2: Array<f64, D>) -> TwoLayerNet<D>
        where D: Dimension{
        return TwoLayerNet{w1, b1, w2, b2};
    }
}