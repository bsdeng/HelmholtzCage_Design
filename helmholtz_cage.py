import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np


class CageOpt:

    def __init__(self):
        self.u0 = 4*np.pi*10**-7  # copper density [kg/m3]
        self.pi = np.pi
        self.gamma = 0.5445

    def opt_formulation(self):
        # Define the list of variables and parameters
        n = cp.Variable(pos=True)  # number of turns
        i = cp.Variable(pos=True)  # current flowing the coil
        a = cp.Variable(pos=True)  # half length of the cage
        u0 = cp.Constant(self.u0)  # u0
        pi = cp.Constant(self.pi)
        gamma = cp.Constant(self.gamma)  # ratio
        # Declare objective function
        objective = n*i*(a**-1)*(4*u0)/(pi*((1+gamma)**2)*cp.sqrt(2+gamma**2))
        obj = cp.Minimize(objective)
        # Declare constraints
        con = []
        con += [n*i*(a**-1)*(4*u0)/(pi*((1+gamma)**2)*cp.sqrt(2+gamma**2)) >= 0.0002]
        con += [i <= 9.3]
        con += [n >= 30]
        con += [a <= 1]
        # Formulate and solve the optimization problem
        pro = cp.Problem(obj, con)
        pro.solve(gp=True)
        print("Optimal value: ", pro.value)
        print("Cage length: ", 2*a.value)
        print("Cage current: ", i.value)
        print("Coil number of turns", n.value)
        return pro


if __name__ == '__main__':
    # Estimates the magnetic dipole
    Cage = CageOpt()
    Cage.opt_formulation()

"""
class HelmCage:
    def __init__(self):
        self.n = 20
        self.u0 = 4*np.pi*10**-7
        self.i = 8
        self.l = 2
        self.d = 2*0.5445

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
    helm_cage = HelmCage()
    helm_cage.main()
"""