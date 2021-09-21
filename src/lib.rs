use std::time::Instant;

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use finitediff::*;
use ndarray::{array, Ix1, Ix2, s};
use rand::prelude::*;

type Vecf = ndarray::Array<f64, Ix1>;
type VecfView<'a> = ndarray::ArrayView<'a, f64, Ix1>;

type Matx = ndarray::Array2<f64>;
type MatxfView<'a> = ndarray::ArrayView<'a, f64, Ix2>;


#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn length_calculator() {
        let q = array![0.0,1.0,0.0];
        assert!(quaternion_length(q.view()) - 1 <= 0.00000000001)
    }
}


fn quaternion_length(q: VecfView) -> f64 {
    ((1.0 - q.dot(&q).sqrt()).powi(2)).sqrt()
}

fn cross_product(a: VecfView, b: VecfView) -> Vecf {
    array![
    (a[1]*b[2]-a[2]*b[1]),
    -(a[0]*b[2]-a[2]*b[0]),
    (a[0]*b[1]-a[1]*b[0])]
}

fn rotate_vec_by_quat(v: VecfView, q: VecfView) -> Vecf {
    let u = array![q[0], q[1], q[2]];
    let u = u.view();
    let s = q[3];
    2.0 * u.dot(&v) * &u + (s * s - u.dot(&u)) * &v + 2.0 * s * cross_product(u, v)
}