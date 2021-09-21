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
        assert!(quaternion_length(q.view()) - 1.0 <= 0.00000000001)
    }
    #[test]
    fn cross() {
        let q = array![1.0, 2.0, 3.0];
        let u = array![1.0, 5.0, 12.0];
        let c = cross_product(q.view(), u.view());
        let dif = c - array![9.0, -9.0, 3.0];
        assert!(dif.dot(&dif).abs() <=0.00001);
    }

    #[test]
    fn rotation() {
        let v = array![2.0, 3.0,5.0];
        let q = array![0.25,0.25,0.0,0.0];
        let q = normalize_quaternion(q.view());
        let expected = array![2.9999999999999996, 2.0, -4.999999999999999];
        let dif = rotate_vec_by_quat(v.view(), q.view())-expected;
        let len_dif = dif.dot(&dif).abs();
        assert!(len_dif <= 0.0000000001);
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

fn normalize_quaternion(q: VecfView) -> Vecf {
    q.to_owned() / (q.dot(&q)).sqrt()
}

fn normalize_qd(qd: VecfView) -> Vecf {
    let q = qd.slice(s![..4]);
    let d = qd.slice(s![4..]);
    let q = normalize_quaternion(q);
    array![q[0], q[1], q[2], q[3], d[0], d[1], d[2]]
}

fn rotate_vec_by_quat(v: VecfView, q: VecfView) -> Vecf {
    let u = array![q[0], q[1], q[2]];
    let u = u.view();
    let s = q[3];
    2.0 * u.dot(&v) * &u + (s * s - u.dot(&u)) * &v + 2.0 * s * cross_product(u, v)
}

