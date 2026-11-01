"""Independent numerical checks against direct least squares and scalar search."""
from itertools import combinations
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from fitting import solve_exact, fitted_values
from minlp_benchmark import parameter_bounds

set_num_threads(1)
threadpool_limits(limits=1)


def direct_sse(x,Y,z,support):
    u=(x-x[0])/(x[-1]-x[0]); knot=(z-x[0])/(x[-1]-x[0])
    line=np.column_stack([np.ones(len(x)),u])
    hinge=np.column_stack([line,np.maximum(0,u-knot)])
    total=0.0
    for j,y in enumerate(Y):
        design=hinge if j in support else line
        residual=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        total+=residual@residual
    return float(total)


def oracle(x,Y,k):
    """Exhaustive supports; dense grid plus local scalar refinement (no solver formulas)."""
    points=np.unique(np.r_[np.linspace(x[1],x[-2],241),x[1:-1]])
    best=np.inf
    for support in combinations(range(len(Y)),k):
        f=lambda z:direct_sse(x,Y,z,support)
        values=np.array([f(z) for z in points]); best=min(best,values.min())
        for i in range(1,len(points)-1):
            if values[i]<=values[i-1] and values[i]<=values[i+1]:
                ans=minimize_scalar(f,bounds=(points[i-1],points[i+1]),method='bounded',options={'xatol':1e-12})
                best=min(best,ans.fun)
    return best


class MethodologyTests(unittest.TestCase):
    def test_exhaustive_support_oracle(self):
        for seed in range(5):
            rng=np.random.default_rng(seed)
            x=np.cumsum(rng.uniform(.2,2,9)); Y=rng.normal(size=(5,9))
            for k in [0,1,2,5]:
                with self.subTest(seed=seed,k=k):
                    result=solve_exact(x,Y,k)
                    self.assertAlmostEqual(result['optimal_loss'],oracle(x,Y,k),places=7)
                    self.assertAlmostEqual(result['optimal_loss'],np.sum((Y-fitted_values(x,result))**2),places=7)

    def test_dense_equals_sparse_k_J(self):
        rng=np.random.default_rng(51); x=np.arange(10.)
        Y=rng.normal(size=(6,10))
        a,b=solve_exact(x,Y),solve_exact(x,Y,6)
        self.assertAlmostEqual(a['optimal_loss'],b['optimal_loss'],places=8)

    def test_ties_constants_and_coincident_crossings(self):
        x=np.arange(10.); base=np.sin(x)
        Y=np.vstack([base,-base,base,np.ones(10),2+3*x])
        for k in [0,1,2,3,5]:
            a=solve_exact(x,Y,k)
            self.assertAlmostEqual(a['optimal_loss'],oracle(x,Y,k),places=7)

    def test_repeated_times_keep_distinct_readings(self):
        x=np.array([0.,1.,2.,2.,3.,4.,5.,6.,7.])
        Y=np.random.default_rng(6).normal(size=(4,9))
        for k in [1,2,4]:
            a=solve_exact(x,Y,k)
            self.assertAlmostEqual(a['optimal_loss'],oracle(x,Y,k),places=7)

    def test_affine_coordinates_and_response_offsets(self):
        x=np.arange(12.); Y=np.random.default_rng(7).normal(size=(4,12))
        a=solve_exact(x,Y,2); b=solve_exact(2000+3*x,Y+100000,2)
        self.assertAlmostEqual(a['optimal_loss'],b['optimal_loss'],places=7)
        np.testing.assert_allclose(fitted_values(2000+3*x,b)-100000,fitted_values(x,a),atol=1e-7)

    def test_noiseless_interior_knot(self):
        x=np.arange(12.); z=4.37
        Y=np.vstack([1+.2*x+2*np.maximum(0,x-z),2-.3*x,3-.2*x-3*np.maximum(0,x-z)])
        a=solve_exact(x,Y,2)
        self.assertAlmostEqual(a['shared_breakpoint'],z,places=7)
        self.assertLess(abs(a['optimal_loss']),1e-7)

    def test_big_m_and_bounds(self):
        for x in [np.arange(-4.,5.),np.array([-4.,-3.,-2.,-2.,0.,1.,2.,3.,4.])]:
            Y=np.random.default_rng(2).normal(size=(3,len(x))); b=parameter_bounds(x,Y)
            ii,jj=np.triu_indices(len(x),1); valid=x[jj]>x[ii]; ii,jj=ii[valid],jj[valid]
            slopes=(Y[:,jj]-Y[:,ii])/(x[jj]-x[ii])
            np.testing.assert_allclose(b['lm'],slopes.min(axis=1)); np.testing.assert_allclose(b['um'],slopes.max(axis=1))
            for slope in [b['lm'],b['um']]:
                for intercept in [b['lc'],b['uc']]:
                    affine=slope[:,None]*x+intercept[:,None]
                    self.assertTrue(np.all(affine>=b['fit_lb']-1e-12))
                    self.assertTrue(np.all(affine<=b['fit_ub']+1e-12))
                    self.assertTrue(np.all(abs(affine-b['fit_lb'])<=b['M1']+1e-12))
            self.assertTrue(np.all(x<=x[1]+b['M2']+1e-12))
            self.assertTrue(np.all(x>=x[-2]-b['M3']-1e-12))


if __name__=='__main__':
    unittest.main()
