__author__ = "Austin W. Milne"
__credits__ = ["Austin W. Milne", "Japmeet Brar", "Joshua Selvanayagam", "Kevin Chu"]
__email__ = "awbmilne@uwaterloo.ca"
__version__ = "1.0"
__date__ = "March 8, 2022"

"""
This code was written for Project #1 of ME303 "Advanced Engineering Mathmatics" in the Winter 2022 term.
The goal is to numerically solve an ODE relating to the occilation of a collapsing vacuum bubble in
water. There are a number of solution methods implemented and compared. In some cases, these solutions
are also compared to the analytical solution.
"""

import copy
import math
import numpy as np
from sympy import E, Matrix
from sympy.matrices import Matrix

class Solvy_boi:  #(*@\label{code:solver_init_start}@*)
    """ 
    ODE numerical solution class.
    This Class takes a system of first order ODEs and provides a number of
    methods available for solving the system numerically.
    """
    def __init__(self, symbol, symbols, functions, s_lim, analytical):
        # Store the ODE system
        self.symbol = symbol
        self.symbols = symbols
        self.functions = functions
        self.s_lim = s_lim
        self.analytical = analytical #(*@\label{code:solver_init_end}@*)
    
    # Exclusive Euler solution methodology (*@\label{code:euler_start}@*)
    def e_eul(self, state, dt, t):
        # Increment the positions using Explicit Euler method
        v_subs_dict = list(zip(self.symbols, state)) # Dictionary for value substitution
        slopes = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions]) # Solve for slopes
        new_state = state + dt * slopes # Apply slopes to state
        # Check for sign inversion
        return self.sign_inversion_correction(state, new_state, t) # Return the corrected state (*@\label{code:euler_end}@*)
    
    # Exclusive Runge-Kutta (RK2) solution methodology (*@\label{code:rk2_start}@*)
    def e_rk2(self, state, dt, t):
        # Increment the state using Runge-Kutta (RK2) method
        state_0 = copy.copy(state)
        # Solve for k1
        v_subs_dict = list(zip(self.symbols, state))
        k1 = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions])
        # Solve for k2 using k1
        new_state = state_0 + 0.5 * dt * k1
        v_subs_dict = list(zip(self.symbols, state))
        k2 = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions])
        # Apply the k2 slope to the function
        new_state = state_0 + dt * k2
        # Check for sign inversion
        return self.sign_inversion_correction(state, new_state, t) # Return the corrected state (*@\label{code:rk2_end}@*)

    # Exclusive Runge-Kutta (RK4) solution methodology (*@\label{code:rk4_start}@*)
    def e_rk4(self, state, dt, t):
        # Increment the state using Runge-Kutta (RK4) method
        state_0 = copy.copy(state)
        # Solve for k1
        v_subs_dict = list(zip(self.symbols, state))
        k1 = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions]) #(*@\label{code:np_clip_call}@*)
        # Solve for k2 using k1
        new_state = state_0 + 0.5 * dt * k1
        v_subs_dict = list(zip(self.symbols, state))
        k2 = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions])
        # Solve for k3 using k2
        new_state = state_0 + 0.5 * dt * k2
        v_subs_dict = list(zip(self.symbols, state))
        k3 = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions])
        # Solve for k4 using k3
        new_state = state_0 + dt * k3
        v_subs_dict = list(zip(self.symbols, state))
        k4 = Matrix([np.clip(f.subs(v_subs_dict),-self.s_lim,self.s_lim) for f in self.functions])
        # Apply the k slopes to the function
        new_state = state_0 + dt * (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Check for sign 
        return self.sign_inversion_correction(state, new_state, t) # Return the corrected state(*@\label{code:rk4_end}@*)

    # Sign inversion correction for vacuum bubble collapse (*@\label{code:sign_inversion_start}@*)
    def sign_inversion_correction(self, state, new_state, t):
        i = self.symbols.index(self.symbol)
        if math.copysign(1, state[i]) != math.copysign(1, new_state[i]):
            if (state[i] != 0) and (new_state[i] != 0):
                # print(f"Collapse @ t={t}") # Debug output
                state[i] = -state[i] # Invert the value of consequence
                return -state # Invert the entire state (reverts value of consquence)
        return new_state # Return the new state if no sign change (*@\label{code:sign_inversion_end}@*)
    
    # Analytical solution incrementor (*@\label{code:analytical_start}@*)
    def anl(self, state, dt, t):
        # Solve given lambda for the specified t
        return [lmb(t) for lmb in self.analytical] # Return the solution list to the supplied lambda list (*@\label{code:analytical_end}@*)
    
    # Solution incrementor function  (*@\label{code:incrementor_start}@*)
    def run_solution(self, method, state_0, d_t ,t):
        state = state_0 # Not necessary, just cleanliness
        data_set = [[0] + [v for v in state_0]] # Store the 0 initial data point
        for i in range(int(t / d_t)): # For every incremental step
            time = d_t*(i+1)
            state = method(self, state, d_t, time) # Update the state using the specified method
            data_set.append([d_t*(i+1)] + [v for v in state]) # Append the data point to the array
        return data_set # Return the list of data (*@\label{code:incrementor_end}@*)