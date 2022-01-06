"""
Written by B. S. Deng January 6th, 2022.

It solves the problem of Helmholtz Cage design.
The optimization problem is subject to the cage size, to the current, and the number of turns.

maximize    Vector_Field
s.t.        current
            number of turns
            Size of the coil

where [i], [n], and [a] are the optimization variables.
This problem can be easily solved via geometric programming.
We use the equations presented on the following references:

[1] E. Cayo, J. Pareja, P. E. R. Arapa, "Design and implementation of a geomagnetic field simulator for small satellites
," Conference: III IAA Latin American Cubesat Workshop, Jan. 2019.
[2] R. C. D. Silva, F. C. Guimaraes, J. V. L. D. Loiola, R. A. Borges, S. Battistini, and C. Cappelletti, "Tabletop
testbed for attitude determination and control of nanosatellites," Journal of Aerospace Engineering, vol. 32, no. 1,
2018.
[3] J. Stevens, "CubeSAT ADCS Validation and Testing Apparatus," Western Michigan University, 2016
[4] N. Theoret, "Attitude Determination Control Testing System," Western Michigan University, 2016
"""

# Import the required libraries
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np


class CageOpt:

    def __init__(self):
        self.u0 = 4*np.pi*10**-7  # permeability of free space
        self.pi = np.pi           # pi cte = 3.14
        self.gamma = 0.5445       # for a uniform vector field, this value should be 0.5445

    def opt_formulation(self):
        """ Define the list of variables and parameters """
        # Variables and parameters must be positive, ie, they must be constructed with the option `pos=True`
        # Decision Variables
        n = cp.Variable(pos=True)  # number of turns
        i = cp.Variable(pos=True)  # current flowing through the coil
        a = cp.Variable(pos=True)  # half-length of the cage
        # Parameter values (both must be positive)
        u0 = cp.Constant(self.u0)
        pi = cp.Constant(self.pi)
        gamma = cp.Constant(self.gamma)
        """ Declare the objective function """
        # the objective function will be the intensity of the magnetic field at the center of the cage
        objective = n*i*(4*u0)/(a*pi*(1+gamma**2)*cp.sqrt(2+gamma**2))
        obj = cp.Minimize(objective)
        """ Declare the constraints """
        # create a blank constraint array
        con = []
        # the minimal intensity at the center of the cage should be greater than 200 uT
        con += [n*i*(4*u0)/(a*pi*(1+gamma**2)*cp.sqrt(2+gamma**2)) >= 0.00015]
        # the maximum current [i] flowing through the coils should be smaller than 9.3 A
        con += [i <= 20]
        # the minimum number of turns [n] should be greater than 30
        con += [n == 20]
        # the half-length of the cage should be not be smaller than 1 m
        con += [a == 1]
        """ Formulate and solve the optimization problem """
        pro = cp.Problem(obj, con)
        pro.solve(gp=True)
        """ Print the obtained results """
        print("Optimal value: ", pro.value)
        print("Cage length: ", 2*a.value)
        print("Cage current: ", i.value)
        print("Coil number of turns", n.value)
        return a.value, i.value, n.value


"""
if __name__ == '__main__':
    # Estimates the magnetic dipole
    Cage = CageOpt()
    a, i, n = Cage.opt_formulation()
"""


class HelmCage:

    def __init__(self, n, i, a):
        self.n = n
        self.u0 = 4*np.pi*10**-7
        self.i = i
        self.l = 2*a
        self.d = 2*0.5445*a

    def ver_cond(self, x, y, z, y_pos, z_pos, i):
        l_dis, d_dis = 0.5*self.l, 0.5*self.d
        av = ( (y-y_pos)**2 + (z-z_pos)**2 )**0.5
        cos1v = (l_dis + x) / ((l_dis+x)**2 + av**2)**0.5
        cos2v = (l_dis - x) / ((l_dis-x)**2 + av**2)**0.5

        b0 = self.n*self.u0*i*(cos2v + cos1v)/(4*np.pi*av)
        z_hat = (y - l_dis)/av
        y_hat = (z - z_pos)/av
        return b0, z_hat, y_hat

    def ver_coil(self, x, y, z):
        # +x & -z
        y_pos, z_pos = +0.5*self.l, -0.5*self.d
        ypzn, z_pn, y_pn = HelmCage.ver_cond(self, x, y, z, y_pos, z_pos, +self.i)
        # +x & +z
        y_pos, z_pos = +0.5*self.l, +0.5*self.d
        ypzp, z_pp, y_pp = HelmCage.ver_cond(self, x, y, z, y_pos, z_pos, +self.i)
        # -x & -z
        y_pos, z_pos = -0.5*self.l, -0.5*self.d
        ynzn, z_nn, y_nn = HelmCage.ver_cond(self, x, y, z, y_pos, z_pos, +self.i)
        # -x & +z
        y_pos, z_pos = -0.5*self.l, +0.5*self.d
        ynzp, z_np, y_np = HelmCage.ver_cond(self, x, y, z, y_pos, z_pos, +self.i)
        # total
        vertical = ypzn + ypzp + ynzn + ynzp
        z = ypzn*z_pn + ypzp*z_pp + ynzn*z_nn + ynzp*z_np
        y = ypzn * y_pn + ypzp * y_pp + ynzn * y_nn + ynzp * y_np
        return vertical, z, y

    def hor_cond(self, x, y, z, x_pos, z_pos, i):
        l_dis, d_dis = 0.5*self.l, 0.5*self.d
        ah = ((x - x_pos) ** 2 + (z - z_pos) ** 2) ** 0.5
        cos1h = (l_dis + y) / ((l_dis + y) ** 2 + ah ** 2) ** 0.5
        cos2h = (l_dis - y) / ((l_dis - y) ** 2 + ah ** 2) ** 0.5

        b0 = self.n*self.u0*i*(cos2h + cos1h)/(4*np.pi*ah)
        z_hat = (x - l_dis)/ah
        x_hat = (z - z_pos)/ah
        return b0, z_hat, x_hat

    def hor_coil(self, x, y, z):
        # +x & -z
        x_pos, z_pos = +0.5*self.l, -0.5*self.d
        xpzn, z_pn, x_pn = HelmCage.hor_cond(self, x, y, z, x_pos, z_pos, +self.i)
        # +x & +z
        x_pos, z_pos = +0.5 * self.l, +0.5 * self.d
        xpzp, z_pp, x_pp = HelmCage.hor_cond(self, x, y, z, x_pos, z_pos, +self.i)
        # -x & -z
        x_pos, z_pos = -0.5 * self.l, -0.5 * self.d
        xnzn, z_nn, x_nn = HelmCage.hor_cond(self, x, y, z, x_pos, z_pos, +self.i)
        # -x & +z
        x_pos, z_pos = -0.5 * self.l, +0.5 * self.d
        xnzp, z_np, x_np = HelmCage.hor_cond(self, x, y, z, x_pos, z_pos, +self.i)
        # total
        horizontal = xpzn + xpzp + xnzn + xnzp
        z = xpzn*z_pn + xpzp*z_pp + xnzn*z_nn + xnzp*z_np
        x = + xpzn*x_pn + xpzp*x_pp + xnzn*x_nn + xnzp*x_np
        return horizontal, z, x

    def main(self):
        b_field = np.zeros([2 * 200 + 1, 2 * 200 + 1])
        x_field = np.zeros([2 * 200 + 1, 2 * 200 + 1])
        y_field = np.zeros([2 * 200 + 1, 2 * 200 + 1])
        z_field = np.zeros([2 * 200 + 1, 2 * 200 + 1])
        for y_ind in range(-200, 200+1, 1):
            y = y_ind/100
            for z_ind in range(-200, 200+1, 1):
                z = z_ind/100
                bh, zh, xh = HelmCage.hor_coil(self, 0, y, z)
                bv, zv, yv = HelmCage.ver_coil(self, 0, y, z)
                x_field[y_ind + 200, z_ind + 200] = xh
                b_field[y_ind+200, z_ind+200] = bh + bv  # B does not have any problem
                z_field[y_ind+200, z_ind+200] = zh#+  #zv  # zh+zv  # horizontal does not show problem,
                y_field[y_ind+200, z_ind+200] = yv  # vertical has a problem
        plt.imshow(b_field, cmap="jet")
        #plt.plot(145.55, 100, marker='o', color='black')
        #plt.plot(145.55, 300, marker='o', color='black')
        #plt.plot(254.45, 100, marker='o', color='black')
        #plt.plot(254.45, 300, marker='o', color='black')
        plt.title('Mag Field')
        plt.colorbar()
        plt.clim(0, 0.0004)
        plt.show()


if __name__ == '__main__':
    Cage = CageOpt()
    a, i, n = Cage.opt_formulation()
    helm_cage = HelmCage(n, i, a)
    helm_cage.main()
