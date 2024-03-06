from scipy.interpolate import CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt

import do_mpc
from casadi import *
import math


class MPCController:

    def __init__(self, timestep=0.1, horizon=5):
        model_type = 'continuous' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # Model Variables
        # State Variables
        self.x_pos = self.model.set_variable(var_type='_x', var_name='x_pos', shape=(1,1))
        self.y_pos = self.model.set_variable(var_type='_x', var_name='y_pos', shape=(1,1))
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
        self.vel = self.model.set_variable(var_type='_x', var_name='vel', shape=(1,1))

        # Input Variables
        self.accel_set = self.model.set_variable(var_type='_u', var_name='accel_set')
        self.steering_set = self.model.set_variable(var_type='_u', var_name='steering_set')

        # Constant for the distance between front and back wheels
        WHEEL_BASE = (1.05234 + 1.4166)/2.0 

        # Define right hand side of equation
        self.model.set_rhs('x_pos', self.vel*cos(self.theta))
        self.model.set_rhs('y_pos', self.vel*sin(self.theta))
        self.model.set_rhs('theta', self.vel*tan(self.steering_set * np.pi/3.0)/WHEEL_BASE)
        self.model.set_rhs('vel', self.accel_set)

        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        # Setup optimizer
        setup_mpc = {
            'n_horizon': horizon,
            't_step': timestep,
            'n_robust': 5,
            'store_full_solution': False,
        }
        self.mpc.set_param(**setup_mpc)

        # define tuning factor for quadratic penalty
        self.mpc.set_rterm(
            accel_set=0.1,
            steering_set=1.0
        )

        # Lower bounds on states:
        self.mpc.bounds['lower','_x', 'x_pos'] = 0.0
        # mpc.bounds['lower','_x', 'y_pos'] = -100.0
        self.mpc.bounds['lower','_x', 'theta'] = -0.5*np.pi
        # self.mpc.bounds['lower','_x', 'vel'] = 1.0
        # # Upper bounds on states
        # mpc.bounds['upper','_x', 'x_pos'] = 100.0
        # mpc.bounds['upper','_x', 'y_pos'] = 100.0
        self.mpc.bounds['upper','_x', 'theta'] = 0.5*np.pi
        # self.mpc.bounds['upper','_x', 'vel'] = 80.0

        # # Lower bounds on inputs:
        self.mpc.bounds['lower','_u', 'accel_set'] = -1.0
        self.mpc.bounds['lower','_u', 'steering_set'] = -1.0
        # # Upper bounds on inputs:
        self.mpc.bounds['upper','_u', 'accel_set'] = 1.0
        self.mpc.bounds['upper','_u', 'steering_set'] = 1.0

        self.mpc.settings.supress_ipopt_output()
        # self.mpc.settings.set_linear_solver(solver_name = "MA27")

        # Scaling
        # self.mpc.scaling['_x', 'theta'] = 2
        # self.mpc.scaling['_u', 'accel_set'] = 2

    def set_objective(self, polar_angle, target_theta, target_vel):
        def interpolate_cubic_curve(point1, slope1, point2, slope2):
            x = np.array([point1[0], point2[0]])
            y = np.array([point1[1], point2[1]])
            dydx = np.array([slope1, slope2])
            
            cubic_spline = CubicHermiteSpline(x, y, dydx, extrapolate=True)
            
            return cubic_spline
        
        RADIUS = 5.0

        target_x = RADIUS * math.cos(polar_angle)
        target_y = RADIUS * math.sin(polar_angle)

        # print(target_x, target_y)

        
        cubic_spline = interpolate_cubic_curve((0, 0), 0, (target_x, target_y), math.tan(target_theta))
        cubic_spline_slope = cubic_spline.derivative()
        # Cubic Spline coeffs
        coeffs = cubic_spline.c
        derivative_coeffs = cubic_spline_slope.c

        # Define objective function
        # mterm is the cost at the end of the horizon (time step n)
        mterm = (5.0*(self.y_pos - (
            coeffs[0][0]*(self.x_pos**3) + 
            coeffs[1][0]*(self.x_pos**2) + 
            coeffs[2][0]*self.x_pos + 
            coeffs[3][0]))**2 + 
            5.0*(self.vel - target_vel)**2 + 
            (self.theta - atan(
            derivative_coeffs[0][0]*(self.x_pos**2) + 
            derivative_coeffs[1][0]*(self.x_pos**1) + 
            derivative_coeffs[2][0]*self.x_pos))**2)
        # lterm is the cost used before the end of horzine (time step < n)
        lterm = ((self.y_pos - (
            coeffs[0][0]*(self.x_pos**3) + 
            coeffs[1][0]*(self.x_pos**2) + 
            coeffs[2][0]*self.x_pos + 
            coeffs[3][0]))**2 +
            (2**(-self.vel))
            # + (self.theta - atan(
            # derivative_coeffs[0][0]*(self.x_pos**2) + 
            # derivative_coeffs[1][0]*(self.x_pos**1) + 
            # derivative_coeffs[2][0]*self.x_pos))**2
            )
        
        self.temp = cubic_spline
        
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.setup()

        return target_x, target_y



