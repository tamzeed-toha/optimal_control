import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import scipy as sc

from sympy import symbols, Eq, solve, simplify, MatrixSymbol
from pprint import pprint
from IPython.display import display, Math


class system:
    def __init__(self, a, b, L=None, silent=True):
        self.value_a = a
        self.value_b = b
        
        self.n_states = a.shape[0]
        self.p_inputs = b.shape[1]
        
        self.x = sp.MatrixSymbol('x', self.n_states, 1)
        self.u = sp.MatrixSymbol('u', 1, self.p_inputs)
        self.t = symbols('t')


        self.A = sp.MatrixSymbol('A', self.n_states, self.n_states)
        self.B = sp.MatrixSymbol('B', self.n_states, self.p_inputs)
        # self.u_star = sp.MatrixSymbol('u^*', 1, self.p_inputs)
        
        self.S = sp.MatrixSymbol('S', self.n_states, self.n_states)
        self.Q = sp.MatrixSymbol('Q', self.n_states, self.n_states)
        self.R = sp.MatrixSymbol('R', self.p_inputs, self.p_inputs, )
        
        self.k = symbols('k', integer=True)
        self.N = symbols('N', integer=True)
        self.x_0 = sp.MatrixSymbol('x_0', self.n_states, 1)
        self.x_N = sp.MatrixSymbol('x_N', self.n_states, 1)
        self.lmda_N = sp.MatrixSymbol('lambda_N', self.n_states, 1)
        
        self.lmda = sp.MatrixSymbol('lambda', self.n_states, 1)
        
        self.Gn = sp.MatrixSymbol('G_n', self.n_states, self.n_states)
        self.r_N = sp.MatrixSymbol('r_N', self.n_states, 1)
        self.P = sp.MatrixSymbol('P', self.n_states, self.n_states)
        
        self.xdot = sp.MatrixSymbol('x_dot', self.n_states, 1)        
        self.lmda_dot = sp.MatrixSymbol('lambda_dot', self.n_states, 1)
        
        if not silent:
            print("system dynamics:")
            display(Math(r'\dot{x} = ' + sp.latex(self.f(self.x, self.u))))
        self.phi = .5 * self.x_N.T @ self.S @ self.x_N
        if L is None:
            self.L = .5 * self.x.T @ self.Q @ self.x + .5 * self.u.T @ self.R @ self.u
        else:
            self.L = L
        self.path_cost = sp.Sum(self.L, (self.k, 0, self.N-1))
        
        if not silent: 
            display(Math(r'\phi = ' + sp.latex(self.phi)))
            display(Math(r'L = ' + sp.latex(self.L)))
            print("path cost:")
            display(Math(r'J_{0} = ' + sp.latex(self.path_cost)))
            
        self.hamiltonian = self.hamiltonian(self.x, self.u, self.lmda)
        self.state_eqn = self.state_eqn_f(self.x, self.u)
        
        if not silent:
            print("Hamiltonian:")
            display(Math(r'H = ' + sp.latex(self.hamiltonian)))
            print("State equation:")
            display(Math(r'\dot{x} = ' + sp.latex(self.state_eqn)))
            
        # manual part: (for now)    
        self.costate_eqn = self.costate_eqn_f(self.x, self.u, self.lmda)
        self.lmda_dot = -self.Q @ self.x - self.A.T @ self.lmda
        self.stationarity_eqn = self.stationarity_eqn_f(self.x, self.u, self.lmda)
        
        ## manual part: (for now)
        self.u_star = -self.R**-1 * self.B.T * self.lmda
        
        if not silent:
            print("Costate equation:")
            display(Math(r'\dot{\lambda} = ' + sp.latex(self.costate_eqn)))
            display(Math(r'\dot{\lambda} = ' + sp.latex(self.lmda_dot)))

            print("Stationarity equation:")
            display(Math(r'\frac{\partial H}{\partial u} = 0 = ' + sp.latex(self.stationarity_eqn)))
            
            print("Optimal control:")
            display(Math(r'u^* = ' + sp.latex(self.u_star)))

    
    def f(self, x, u):
        # if not isinstance(x, sp.Matrix) or not isinstance(x, sp.Function):
        #     x = sp.Matrix(x).reshape(self.n_states, 1)
        # if not isinstance(u, sp.Matrix):
        #     u = sp.Matrix(u).reshape(self.p_inputs, 1)
        # return self.value_a @ x + self.value_b @ u
        return self.A * x + self.B * u

    def hamiltonian(self, x, u, lmda):
        return self.L + lmda.T @ self.f(x, u)
    
    
    def state_eqn_f(self, x, u):
        return Eq(self.xdot, self.f(x, u))
    
    def costate_eqn_f(self, x, u, lmda):
        expr = Eq(self.lmda_dot, -sp.diff(self.hamiltonian, x))
        expr = expr.subs(self.Q.T, self.Q)
        return sp.factor_terms(expr)
    
    def stationarity_eqn_f(self, x, u, lmda):
        expr = sp.diff(self.hamiltonian, u)
        expr = expr.subs(self.R.T, self.R)
        return sp.factor_terms(expr)
    
    
    def fixed_final_optimal_control(self, Q, R, S, x0, rN, N, plot=True, dt=1):
        print("the optimal input for the fixed final time problem is:")
        const_1 = self.R**-1 * self.B.T 
        var_1 = sp.exp(self.A.T * (self.N - self.t))
        const_2 = self.Gn ** -1 * (self.r_N - self.A**N * self.x_0)

        display(Math(r'u^* = ' + sp.latex(const_1) + sp.latex(var_1) + sp.latex(const_2)))
        
        # first we calculate G_n
        r_inv = sp.Matrix(sc.linalg.inv(np.array(R).astype(np.float64)))
        def p_dynamics(pval, k):
            pval = pval.reshape(self.n_states, self.n_states)
            pnot = self.value_a @ pval + pval @ self.value_a.T + self.value_b @ r_inv @ self.value_b.T
            return np.array(pnot).astype(np.float64).flatten()
        
        pvals = sc.integrate.odeint(p_dynamics, np.zeros(self.n_states**2), np.arange(N+1))
        pvals = pvals.reshape(N+1, self.n_states, self.n_states)
        
        # plt.plot(pvals[:,0,0])
        # plt.plot(pvals[:,1,1])
        # plt.plot(pvals[:,0,1])
        # plt.plot(pvals[:,1,0])
        
        # print(pvals)
                
        Gn = pvals[-1]
        Gn_inv = sc.linalg.inv(Gn)
        
        
        const_1 = const_1.subs(self.R, R).subs(self.B, self.value_b)
        const_1 = np.array(const_1).astype(np.float64)
        const_2 = Gn_inv @ (rN - sc.linalg.expm(self.value_a * N) @ x0)
        const_2 = np.array(const_2).astype(np.float64)
        
        
        ustars = []
        for t in np.arange(0, N+1, dt):
            ustar = const_1 @ sc.linalg.expm(self.value_a.T * (N - t)) @ const_2
            ustars.append(ustar)
        
        ustars = np.array(ustars).reshape(N+1, self.p_inputs)

        def f_dynamics(x, t):
            x = x.reshape(self.n_states, 1)
            u = const_1 @ sc.linalg.expm(self.value_a.T * (N - t)) @ const_2
            value = self.value_a @ x + self.value_b @ u 
            return np.array(value).astype(np.float64).flatten()
        x0 = np.array(x0).astype(np.float64).flatten()
        xvals = sc.integrate.odeint(f_dynamics, x0, np.arange(0, N+1, dt))
        xvals = xvals.reshape(N+1, self.n_states)
        
 
            
        cost = [u.T @ np.array(R).astype(np.float64) @ u for u in ustars]
        cost = np.array(cost).sum() * .5

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].plot(xvals)
            ax[0].set_title("State Trajectory")
            ax[0].axhline(rN[0], color="red", linestyle="--")
            ax[1].plot(ustars)
            ax[1].set_title("Control Trajectory")
            
            plt.suptitle(f"Cost: {cost.round(2)}")
            plt.show()
            
        return xvals, ustars, cost
        
        
        
        
        
    def free_final_optimal_control(self, Q, R, S, x0, rN, N, plot=True, dt=1):
        
        def s_dynamics(sval, t):
            sval = sval.reshape(self.n_states, self.n_states)
            sdot = -self.value_a.T @ sval - sval @ self.value_a + sval @ self.value_b @ sc.linalg.inv(np.array(R).astype(np.float64)) @ self.value_b.T @ sval - np.array(Q).astype(np.float64)
            return np.array(sdot).astype(np.float64).flatten()
        
        S = np.array(S).astype(np.float64)
        
        svals = sc.integrate.odeint(s_dynamics, S.flatten(), np.arange(N+1, 0, -dt))
        svals = svals[::-1]   # because s is calculated from reverse
        
        def get_s_val(t):
            values = []
            for i in range(S.size):
                v = np.interp(t, np.arange(0, N+1, dt), svals[:, i])
                values.append(v)
            values = np.array(values).reshape(self.n_states, self.n_states)
            return values
        
        K_vals = []
        for t in np.arange(0, N+1, dt):
            sval = get_s_val(t)
            kval = -sc.linalg.inv(np.array(R).astype(np.float64)) @ self.value_b.T @ sval
            K_vals.append(kval)
        K_vals = np.array(K_vals)
        
        def f_dynamics(x, t):
            x = x.reshape(self.n_states, 1)
            K = sc.linalg.inv(np.array(R).astype(np.float64)) @ self.value_b.T @ get_s_val(t)
            u = -K @ x
            xdot = self.value_a @ x + self.value_b @ u
            return np.array(xdot).astype(np.float64).flatten()
        
        x0 = np.array(x0).astype(np.float64).flatten()
        xvals = sc.integrate.odeint(f_dynamics, x0, np.arange(0, N+1, dt))
        xvals = xvals.reshape(N+1, self.n_states)
        
        ustars = []
        for i,kval in enumerate(K_vals):
            ustar = -kval @ xvals[i]
            ustars.append(ustar)
        ustars = np.array(ustars).reshape(N+1, self.p_inputs)

        xn = xvals[-1]
        final_cost = xn.T @ S @ xn 
        pos_cost = [x.T @ np.array(Q).astype(np.float64) @ x for x in xvals]
        input_cost = [u.T @ np.array(R).astype(np.float64) @ u for u in ustars]
        total_cost = final_cost + .5 * (np.sum(pos_cost) + np.sum(input_cost))
        
        
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].plot(xvals)
            ax[0].set_title("State Trajectory")
            ax[1].plot(ustars)
            ax[1].set_title("Control Trajectory")
            ax[2].plot(svals)
            ax[2].set_title("Costate Trajectory")
            plt.suptitle(f"Cost: {total_cost.round(2)}")
            plt.show()
            
            
        return xvals, ustars, svals, total_cost