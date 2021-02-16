# Dynamic Systems to run tests on.

def vdp_system(x, _t): # Van der Pol Oscillator
    x1, x2 = x
    return [x2,  -x1 + 1.0 * (1 - x1**2) * x2]

def toggle_system(x, _t, alpha1=2.5, alpha2=1.5, n1=1.4, n2=1.1, delta=0.25): # Togle switch, parameters default to those of the 2017 ACC SILL paper.
    x1, x2 = x
    return [alpha1 / (1 + x2**n1) - delta * x1, alpha2 / (1 + x1**n2) - delta * x2]

def lv_system(x, _t): # Predator-Prey equations.
    x1, x2 = x
    prod = x1 * x2
    return [1.1*x1 - .5 * prod, .1 * prod - 0.2 * x2]

def duffing_system(x, _t): # Duffing oscillator
    x1, x2 = x
    return [x2, x1 - x1**3]

def glycol_oscillator(y, _t, J=2.5, A=4, N=1, K1=.52, kap=13, phi=.1,
                      q=4, k=1.8, k1=100, k2=6, k3=16, k4=100, k5=1.28, 
                      k6=12): # 7 dim glycolic oscillator with parameters defaulting to those used in the original deepDMD paper.
    dy0 = J - (k1*y[0]*y[5]/(1+(y[5]/K1)**q))
    dy1 = 2*(k1*y[0]*y[5]/(1+(y[5]/K1)**q)) - k2*y[1]*(N-y[4]) - k6*y[1]*y[4]
    dy2 = k2*y[1]*(N-y[4]) - k3*y[2]*(A-y[5])
    dy3 = k3*y[2]*(A-y[5]) - k4*y[3]*y[4]-kap*(y[3]-y[6])
    dy4 = k2*y[1]*(N-y[4]) - k4*y[3]*y[4] - k6*y[1]*y[4]
    dy5 = -2*(k1*y[0]*y[5]/(1+(y[5]/K1)**q)) + 2*k3*y[2]*(A-y[5]) - k5*y[5]
    dy6 = phi*kap*(y[3]-y[6]) - k*y[6]
    return [dy0, dy1, dy2, dy3, dy4, dy5, dy6]