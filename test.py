import quat_optimiz as opt
import numpy as np 
if __name__ == "__main__":
    inv = np.array([[1.0,2.0,3.0], [3.0,4.404,50.12], [-12.0, 4.0, 12.0]])
    outv = np.array([[1.0,2.0,3.0], [3.0,4.404,50.12], [-12.0, 4.0, 12.0]])
    qd_guess = np.array([0.0,0.0,0.0,0.14,0.0,0.0,0.0])
    max_iter = 100
    print(opt.run_optimization(inv, outv, qd_guess, max_iter))