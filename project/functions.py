import numpy as np  
import sympy as sym


def f(state, u):
    x2, x1, y2, y1, theta3, theta2, theta1, phi2, phi1 = state
    F, freq, tau = u
        
    c2 = 10
    c1 = 10
    c3 = 100
    c4 = 1
    c5 = 1
    c6 = 1
    c7 = 1
        
    x2_dot = -c1 * x2 + F * np.cos(phi1)
    x1_dot = x2
    y2_dot = -c2 * y2 + F * np.sin(phi1)
    y1_dot = y2
    theta3_dot = -c3 * theta3 - c4 * theta2 - c5 * theta1 + freq
    theta2_dot = theta3
    theta1_dot = theta2
    phi2dot = -c6 * phi2 - c7 * phi1 + theta1 + tau
    phi1dot = phi2
    
    return x2_dot, x1_dot, y2_dot, y1_dot, theta3_dot, theta2_dot, theta1_dot, phi2dot, phi1dot





class SystemAgent2D:
    def __init__(self):
        self.x2, self.x1, self.y2, self.y1, self.theta3, self.theta2, self.theta1, self.phi2, self.phi1 = sym.symbols('x2 x1 y2 y1 theta3 theta2 theta1 phi2 phi1')
        self.F, self.u_freq, self.tau = sym.symbols('F u_freq tau')
        self.lmda1, self.lmda2, self.lmda3, self.lmda4, self.lmda5, self.lmda6, self.lmda7, self.lmda8, self.lmda9 = sym.symbols('lambda1 lambda2 lambda3 lambda4 lambda5 lambda6 lambda7 lambda8 lambda9')
        self.t, self.s = sym.symbols('t s')
        
        
        self.c2 = 10
        self.c1 = 10
        self.c3 = 100
        self.c4 = 1
        self.c5 = 1
        self.c6 = 1
        self.c7 = 1
        
        states = [self.x2, self.x1, self.y2, self.y1, self.theta3, self.theta2, self.theta1, self.phi2, self.phi1]
        inputs = [self.F, self.u_freq, self.tau]
        lmdas = [self.lmda1, self.lmda2, self.lmda3, self.lmda4, self.lmda5, self.lmda6, self.lmda7, self.lmda8, self.lmda9]
        self.states = sym.Matrix(states)
        self.u = sym.Matrix(inputs)
        self.lmdas = sym.Matrix(lmdas)
        
        x2_dot = -self.c1 * self.x2 + self.F * sym.cos(self.phi1)
        x1_dot = self.x2
        y2_dot = -self.c2 * self.y2 + self.F * sym.sin(self.phi1)
        y1_dot = self.y2
        theta3_dot = -self.c3 * self.theta3 - self.c4 * self.theta2 - self.c5 * self.theta1 + self.u_freq
        theta2_dot = self.theta3
        theta1_dot = self.theta2
        phi2_dot = -self.c6 * self.phi2 - self.c7 * self.phi1 + self.theta1 + self.tau
        phi1_dot = self.phi2
        x_dot = [x2_dot, x1_dot, y2_dot, y1_dot, theta3_dot, theta2_dot, theta1_dot, phi2_dot, phi1_dot]
        self.x_dot = sym.Matrix(x_dot)
        
        self.A, self.B = self.linearize()
        
    def f(self, state, u):
        subs = {self.x2: state[0], self.x1: state[1], self.y2: state[2], self.y1: state[3], self.theta3: state[4], self.theta2: state[5], self.theta1: state[6], self.phi2: state[7], self.phi1: state[8], self.F: u[0], self.u_freq: u[1], self.tau: u[2]} 
        return np.array(self.x_dot.subs(subs)).astype(np.float64).reshape(-1)


    def linearize(self):
        A = self.x_dot.jacobian(self.states).subs({self.F: 0, self.u_freq: 0, self.tau: 0})
        B = self.x_dot.jacobian(self.u)
        # small angle approximation
        B = B.subs({sym.sin(self.phi1): self.phi1, sym.cos(self.phi1): 1})
        return A, B
    
    def linearized_f(self, state, u):
        if isinstance(state, list):
            state = np.array(state).astype(np.float64).reshape(-1)
        if isinstance(u, list):
            u = np.array(u).astype(np.float64).reshape(-1)
        state_sub = {self.x2: state[0], self.x1: state[1], self.y2: state[2], self.y1: state[3], self.theta3: state[4], self.theta2: state[5], self.theta1: state[6], self.phi2: state[7], self.phi1: state[8]}
        A = np.array(self.A.subs(state_sub)).astype(np.float64)
        B = np.array(self.B.subs(state_sub)).astype(np.float64)
        value = A @ state + B @ u
        return np.array(value).astype(np.float64).reshape(-1)
        

























# def follow_sample_theta_trajectory(state, t, timeseries, phi_ref):
#     x2, x1, y2, y1, theta3, theta2, theta1 = state

#     F,u_freq, tau = sample_control(state, t, timeseries, phi_ref)    
#     df = f(state, (F, u_freq, tau))
#     return df

# def sample_control(state, t, timeseries, phi_ref):
#     x2, x1, y2, y1, theta3, theta2, theta1 = state
#     F = 10
#     u_freq = 2
#     phi_r = np.interp(t, timeseries, phi_ref)
#     tau = (phi_r - theta1) * .5
#     return F, u_freq, tau
    

# def f(state, u):
#     x2, x1, y2, y1, theta3, theta2, theta1 = state
#     F, u_freq, tau = u
        
#     c2 = 10
#     c1 = 10
#     c3 = 100
#     c4 = 1
#     c5 = 1
    
#     x2_dot = -c1 * x2 + F * np.cos(theta1 + tau)
#     x1_dot = x2
#     y2_dot = -c2 * y2 + F * np.sin(theta1 + tau)
#     y1_dot = y2
#     theta3_dot = -c3 * theta3 - c4 * theta2 - c5 * theta1 + u_freq
#     theta2_dot = theta3
#     theta1_dot = theta2
    
#     return x2_dot, x1_dot, y2_dot, y1_dot, theta3_dot, theta2_dot, theta1_dot
