import numpy as np  

def follow_sample_theta_trajectory(state, t, timeseries, phi_ref):
    x2, x1, y2, y1, theta3, theta2, theta1 = state

    F, tau = sample_control(state, t, timeseries, phi_ref)    
    df = f(state, (F, tau))
    return df

def sample_control(state, t, timeseries, phi_ref):
    x2, x1, y2, y1, theta3, theta2, theta1 = state
    F = 10
    phi_r = np.interp(t, timeseries, phi_ref)
    tau = (phi_r - theta1) * 1
    return F, tau
    

def f(state, u):
    x2, x1, y2, y1, theta3, theta2, theta1 = state
    F, tau = u
        
    c2 = 10
    c1 = 10
    c3 = 1.2
    c4 = 1
    
    x2_dot = -c1 * x2 + F * np.cos(theta2 + theta1)
    x1_dot = x2
    y2_dot = -c2 * y2 + F * np.sin(theta2 + theta1)
    y1_dot = y2
    theta3_dot = -c3 * theta3 - c4 * theta2 + tau 
    theta2_dot = theta3
    theta1_dot = theta2
    
    return x2_dot, x1_dot, y2_dot, y1_dot, theta3_dot, theta2_dot, theta1_dot