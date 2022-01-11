"""

. /Users/bsdeng/opt/anaconda3/bin/activate && conda activate /Users/bsdeng/opt/anaconda3/envs/star_tracker;
cd /Users/bsdeng/Desktop/code/helmholtz_cage
pylint helmholtz_cage.py
python helmholtz_cage.py

Written by B. S. Deng January 6th, 2022.

It solves the problem of Square-Shaped Helmholtz Cage design.
The optimization problem is subject to the cage size, to the current, and the number of turns.

maximize    Vector_Field
s.t.        current
            number of turns
            Size of the coil

where [i], [n], and [a] are the optimization variables.
This problem can be easily solved via geometric programming.
We use the equations presented on the following references:
"""

# Import the required libraries
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np


class Cage:
    """
    This class contains the parameter values that will be used in the optimization problem.
    ---------
    Atributes
    ---------
    u_0   : float .:. permeability of free space [-]
    gamma : float .:. constant value for uniform vector field [-]
    ub_i  : float .:. current upper bound [A]
    turns : int   .:. number of coil turns [-]
    ub_a  : float .:. half length upper bound [m]
    lb_a  : float .:. half length lower bound [m]
    des_b : float .:. desired magnetic field intensity at the center of the cage [Tesla]
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the cage object.
        """
        self.u_0 = 4*np.pi*10**-7  # permeability of free space [-]
        self.gamma = 0.5445       # for a uniform vector field, this value should be 0.5445 [-]
        self.current_limits = [0, 20]    # allowed current range [A]
        self.turns_limits = [0, 20]      # allowed turns range [-]
        self.length_limits = [1, 1]      # allowed length range [m]
        self.b_limits = [0.00015, 0.00015]     # the desired magnetic field value [Tesla]
    def info_objective(self):
        """
        It prints the info about the class.
        """
        print('This class stores the optimization parameters.')
        print('The intensity at the center of the cage is ranging between ', \
              self.b_limits[0], ' to ', self.b_limits[1], 'Tesla.')
    def info_constraints(self):
        """
        It prints the info about the class.
        """
        print('This class stores the optimization parameters.')
        print('The current is ranging between ', \
              self.current_limits[0], ' to ', self.current_limits[1], 'Tesla.')
        print('The number of turns is ranging between ', \
              self.turns_limits[0], ' to ', self.turns_limits[1], 'Tesla.')
        print('The half-length is ranging between ', \
              self.length_limits[0], ' to ', self.length_limits[1], 'Tesla.') 
        
class OptimalCage:
    def __init__(self, length, distance, current, turns):
        # optimal values
        self.opt_length = length
        self.opt_distance = distance
        self.opt_current = current
        self.opt_turns = turns

def squared_cage_optimization():
    '''
    Formulate and Solve the optimization problem.
    Args:
        max_i = maximum allowed current
        turns = maximum allowed number of turns
        max_a = maximum allowed half-length
        min_a = minimum allowed half-length
        des_b = the desired magnetic field intensity at the center of the cage.
    Returns:
        opt_a = the optimal half-length.
        opt_n = the optimal number of turns.
        opt_i = the optimal current.
    '''
    # if ,,, << ,,,
    # raise ValueEror()
    # optimization variables
    opt_n = cp.Variable(pos=True)  # number of turns
    opt_i = cp.Variable(pos=True)  # current flowing through the coil
    opt_a = cp.Variable(pos=True)  # half-length of the cage
    # constants
    u_0 = cp.Constant(Cage().u_0)
    gamma = cp.Constant(Cage().gamma)
    # objective function
    objective = opt_n*opt_i*(4*u_0)/(opt_a*np.pi*(1+gamma**2)*cp.sqrt(2+gamma**2))
    obj = cp.Minimize(objective)
    # constraints
    con = []
    con += [opt_n*opt_i*4*u_0/(opt_a*np.pi*(1+gamma**2)*cp.sqrt(2+gamma**2)) <= Cage().b_limits[1]]
    con += [opt_n*opt_i*4*u_0/(opt_a*np.pi*(1+gamma**2)*cp.sqrt(2+gamma**2)) >= Cage().b_limits[0]]
    con += [opt_i <= Cage().current_limits[1]]
    con += [opt_n == Cage().turns_limits[1]]
    con += [opt_a <= Cage().length_limits[1]]
    con += [opt_a >= Cage().length_limits[0]]
    # Construct & Solve the optimization problem.
    pro = cp.Problem(obj, con)
    pro.solve(gp=True)
    print("Optimal value: ", pro.value, " Tesla")
    print("Cage Half-Length: ", opt_a.value, " meters")
    print("Cage current: ", opt_i.value, " A")
    print("Coil number of turns", opt_n.value)
    length = 2*opt_a.value
    distance = 2*(Cage().gamma)*opt_a.value
    opt_design = OptimalCage(length, distance, opt_i.value, opt_n.value)
    return opt_design

def vertical_wire(coordinate, y_pos, z_pos):
    '''
    Computes the contribution of n vertical wires at y_pos and z_pos
    '''
    x, y, z = coordinate
    l_dis = 0.5*opt_design.opt_length
    turns = opt_design.opt_turns
    current = opt_design.opt_current   
    av = ( (y-y_pos)**2 + (z-z_pos)**2 )**0.5
    cos1v = (l_dis + x) / ((l_dis+x)**2 + av**2)**0.5
    cos2v = (l_dis - x) / ((l_dis-x)**2 + av**2)**0.5
    b0 = turns*Cage().u_0*current*(cos2v + cos1v)/(4*np.pi*av)
    return b0

def sum_vertical_wires(coordinate):
    '''
    Computes the summation of vertical conductor at 4 different locations (with n wires)
    '''
    pn = vertical_wire(coordinate, +0.5*opt_design.opt_length, -0.5*opt_design.opt_distance)
    pp = vertical_wire(coordinate, +0.5*opt_design.opt_length, +0.5*opt_design.opt_distance)
    nn = vertical_wire(coordinate, -0.5*opt_design.opt_length, -0.5*opt_design.opt_distance)
    np = vertical_wire(coordinate, -0.5*opt_design.opt_length, +0.5*opt_design.opt_distance)
    # total
    summation = pn + pp + nn + np
    return summation

def horizontal_wire(coordinate, x_pos, z_pos):
    '''
    Computes the contribution of n vertical wires at x_pos and z_pos
    '''
    x, y, z = coordinate
    l_dis = 0.5*opt_design.opt_length
    turns = opt_design.opt_turns
    current = opt_design.opt_current    
    ah = ((x - x_pos) ** 2 + (z - z_pos) ** 2) ** 0.5
    cos1h = (l_dis + y) / ((l_dis + y) ** 2 + ah ** 2) ** 0.5
    cos2h = (l_dis - y) / ((l_dis - y) ** 2 + ah ** 2) ** 0.5
    b0 = turns*Cage().u_0*current*(cos2h + cos1h)/(4*np.pi*ah)
    return b0

def sum_horizontal_wires(coordinate):
    '''
    Computes the summation of vertical conductor at 4 different locations (with n wires)
    '''
    xpzn = horizontal_wire(coordinate, +0.5*opt_design.opt_length, -0.5*opt_design.opt_distance)
    xpzp = horizontal_wire(coordinate, +0.5*opt_design.opt_length, +0.5*opt_design.opt_distance)
    xnzn = horizontal_wire(coordinate, -0.5*opt_design.opt_length, -0.5*opt_design.opt_distance)
    xnzp = horizontal_wire(coordinate, -0.5*opt_design.opt_length, +0.5*opt_design.opt_distance)
    # total
    summation = xpzn + xpzp + xnzn + xnzp
    return summation

def main():
    '''
    Compute and save the vector field intensity around the center of the cage.
    '''
    step = 0.01  # 0.01 for centimeter or 0.001 for milimeter
    b_range = [-2, 2]  # in meters
    b_linspace = np.arange(b_range[0], b_range[1]+step, step)
    b_intensity_map = np.zeros([np.size(b_linspace), np.size(b_linspace)])
    coordinate = np.zeros(3)
    for coordinate[1] in b_linspace:
        for coordinate[2] in b_linspace:
            b_from_hor_wires = sum_horizontal_wires(coordinate)  #print(b_from_hor_wires)
            b_from_ver_wires = sum_vertical_wires(coordinate)  #print(b_from_ver_wires)
            plane_index = (int((coordinate[1]+b_range[1])/step), \
                           int((coordinate[2]+b_range[1])/step))
            b_intensity_map[plane_index] = b_from_hor_wires + b_from_ver_wires
    plt.imshow(b_intensity_map, cmap="jet")
    plt.plot(145.55, 100, marker='o', color='black')
    plt.plot(145.55, 300, marker='o', color='black')
    plt.plot(254.45, 100, marker='o', color='black')
    plt.plot(254.45, 300, marker='o', color='black')
    plt.title('Magnetic Field around the center of the cage')
    plt.colorbar()
    plt.clim(0, 0.0004)
    plt.show()
    plt.savefig('cage.png')


if __name__ == '__main__':
    opt_design = squared_cage_optimization()
    main()
