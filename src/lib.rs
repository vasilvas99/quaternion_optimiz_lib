use std::process;

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use finitediff::*;
use ndarray::{array, Ix1, Ix2, s};
use numpy::ToPyArray;
use pyo3::prelude::*;

type Vecf = ndarray::Array<f64, Ix1>;
type VecfView<'a> = ndarray::ArrayView<'a, f64, Ix1>;

type Matx = ndarray::Array2<f64>;
type MatxfView<'a> = ndarray::ArrayView<'a, f64, Ix2>;


#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::prelude::*;

    use crate::*;

    fn normalize_quaternion(q: VecfView) -> Vecf {
        q.to_owned() / (q.dot(&q)).sqrt()
    }

    fn normalize_qd(qd: VecfView) -> Vecf {
        let q = qd.slice(s![..4]);
        let d = qd.slice(s![4..]);
        let q = normalize_quaternion(q);
        array![q[0], q[1], q[2], q[3], d[0], d[1], d[2]]
    }

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
        assert!(dif.dot(&dif).abs() <= 0.00001);
    }

    #[test]
    fn rotation() {
        let v = array![2.0, 3.0,5.0];
        let q = array![0.25,0.25,0.0,0.0];
        let q = normalize_quaternion(q.view());
        let expected = array![2.9999999999999996, 2.0, -4.999999999999999];
        let dif = rotate_vec_by_quat(v.view(), q.view()) - expected;
        let len_dif = dif.dot(&dif).abs();
        assert!(len_dif <= 0.0000000001);
    }

    #[test]
    fn cost_tester() {
        //this test depends on the implementation of the cost. If the cost is calculated in another
        //way, the test should be redone
        let input = ndarray::arr2(&[[1.0, 2.0, 3.0], [12.0, 3.0, -90.0], [-80.0, 12.0, 3.0]]);
        let qd = array![0.25,0.25,0.0,0.0,0.0,0.0,0.0];
        let qd = normalize_qd(qd.view());
        let output = ndarray::arr2(&[[13.0, 25.0, 31.0], [199.0, 23.0, -9.0], [9.0, 12.0, 4.0]]);
        assert!(calculate_cost(&qd, input.view(), output.view()) - 242.30765567765292 <= 0.000000000000001)
    }

    fn generate_optimiz_test_data(qd: VecfView, size: usize, scale: f64) -> (Matx, Matx) {
        let q = qd.slice(s![..4]);
        let d = qd.slice(s![4..]);
        let mut initial_vecs: Matx = ndarray::Array::zeros((size, 3));
        let mut rng = thread_rng();

        for mut row in initial_vecs.rows_mut() {
            row[0] = scale * rng.gen::<f64>();
            row[1] = scale * rng.gen::<f64>();
            row[2] = scale * rng.gen::<f64>();
        }
        let mut output_vecs: Matx = initial_vecs.to_owned();
        for mut row in output_vecs.rows_mut() {
            let new_row = rotate_vec_by_quat(row.view(), q.view()) + d.view();
            row[0] = new_row[0] + rng.gen::<f64>();
            row[1] = new_row[1] + rng.gen::<f64>();
            row[2] = new_row[2] + rng.gen::<f64>();
        }
        (initial_vecs, output_vecs)
    }

    #[test]
    fn bfgs() {
        let quat = array![0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0];
        println!("qd before normalization {}", quat);
        let quat = normalize_qd(quat.view()).to_owned();
        println!("qd after normalization {}", quat);

        let size = 300 as usize;
        let scale = 100.0;

        println!("Starting test data generation");
        let now = Instant::now();
        let (inv, outv) = generate_optimiz_test_data(quat.view(), size, scale);
        println!("Data generation over, time taken: {} ms", now.elapsed().as_millis());

        let guess = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let guess = normalize_qd(guess.view());


        println!("Starting Optimization");
        let now = Instant::now();
        let res = run_bfgs(inv, outv, guess, 100).unwrap();
        println!("Optimization over, time taken {} ms", now.elapsed().as_millis());
        println!("{:#?}", res);
        println!("\n\n========================================\n\nFinal results: \
    \nExpected: {} \n\nCalculated: {}", quat, res.param);
        assert!((quat[3] - res.param[3]).abs() <= 0.1)
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

fn calculate_cost(qd: &Vecf, initial_vecs: MatxfView, target_vecs: MatxfView) -> f64 {
    let q = qd.slice(s![..4]);
    let d = qd.slice(s![4..]);
    let mut cost = 10.0 * quaternion_length(q.view()); //Normalization constraint
    for i in 0..initial_vecs.shape()[0] {
        let res = rotate_vec_by_quat(initial_vecs.row(i).view(), q.view())
            + d.view() - target_vecs.row(i).view();
        cost += res.dot(&res);
    }
    cost.sqrt()
}

#[derive(Debug)]
struct QDProblem {
    input_vectors: Matx,
    output_vectors: Matx,
}


impl ArgminOp for QDProblem {
    type Param = Vecf;
    type Output = f64;
    type Hessian = Matx;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(calculate_cost(&_param, self.input_vectors.view(),
                          self.output_vectors.view()))
    }

    fn gradient(&self, _param: &Self::Param) -> Result<Self::Param, Error> {
        Ok(
            (*_param).forward_diff(&|x|
                calculate_cost(&x, self.input_vectors.view(),
                               self.output_vectors.view()))
        )
    }
}


impl QDProblem {
    pub fn new_qd_finder(input_vectors: Matx, output_vectors: Matx) -> QDProblem {
        QDProblem {
            input_vectors,
            output_vectors,
        }
    }
}

fn run_bfgs(input_vectors: Matx, target_vectors: Matx, qd_guess: Vecf, max_iter: u64)
            -> Result<IterState<QDProblem>, Error> {
    let cost = QDProblem::new_qd_finder(input_vectors, target_vectors);


    // set up a line search
    let line_search = MoreThuenteLineSearch::new().c(1e-6, 0.9)?;

    // Set up solver
    let solver = LBFGS::new(line_search, 12)
        .with_tol_cost(0.00001);

    // Run solver
    let res = Executor::new(cost, solver, qd_guess)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(max_iter)
        .run()?;
    Ok(res.state)
}


#[pymodule]
// Finds the optimal quaternion and distance that can be used to transform the vectors from the
// input set to those from the target set. Uses the L-BFGS algorithm and the initial guess should
// be close enough to the exact result. Avoid q = [0,0,0,0]!!.
fn quat_optimiz(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "run_optimization")]
    fn run_optimization_py<'py>(_py: Python<'py>, input_vectors: &numpy::PyArray2<f64>,
                                target_vectors: &numpy::PyArray2<f64>,
                                qd_guess: &numpy::PyArray1<f64>,
                                max_iter: u64) -> &'py numpy::PyArray1<f64>
    {
        let input_vectors = unsafe { input_vectors.as_array_mut() };
        let input_vectors = input_vectors.to_owned();
        let target_vectors = unsafe { target_vectors.as_array_mut() };
        let target_vectors = target_vectors.to_owned();
        let qd_guess = unsafe { qd_guess.as_array_mut() };
        let qd_guess = qd_guess.to_owned();

        let res = match run_bfgs(input_vectors, target_vectors, qd_guess, max_iter) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("Something went wrong when running the optimization: {}", e);
                process::exit(1);
            }
        };
        res.param.to_pyarray(_py)
    }

    Ok(())
}