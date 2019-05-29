from dmp import DMPs
import numpy as np


class DmpsGpis(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):

        # call super class constructor
        super(DmpsGpis, self).__init__(pattern='discrete', **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.bfs) * self.bfs**1.5 / self.c

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        # desired spacings along x
        # need to be spaced evenly between 1 and exp(-ax)
        # lowest number should be only as far as x gets
        first = np.exp(-self.cs.ax*self.cs.run_time)
        last = 1.05 - first
        des_c = np.linspace(first,last,self.bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # x = exp(-c), solving for c
            self.c[n] = -np.log(des_c[n])

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """

        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return y_des[:,-1].copy()

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray):
            x = x[:,None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        #efficiently calculate weights for BFs using weighted linear regression
        self.w = np.zeros((self.dmps, self.bfs))
        for d in range(self.dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.bfs):
                numer = np.sum(x_track * psi_track[:,b] * f_target[:,d])
                denom = np.sum(x_track**2 * psi_track[:,b])
                self.w[d,b] = numer / (k * denom)

    def step(self, tau=1.0, state_fb=None, external_force=None, contact_judge=None):
        """Run the DMP system for a single timestep.

       tau float: scales the timestep
                  increase tau to make the system execute faster
       state_fb np.array: optional system feedback
        """

        # run canonical system
        cs_args = {'tau':tau,
                   'error_coupling':1.0}
        if state_fb is not None:
            # take the 2 norm of the overall error
            state_fb = state_fb.reshape(1,self.dmps)
            dist = np.sqrt(np.sum((state_fb - self.y)**2))
            cs_args['error_coupling'] = 1.0 / (1.0 + 10*dist)
        x = self.cs.step(**cs_args)

        # generate basis function activation
        psi = self.gen_psi(x)


        for d in range(self.dmps):

            # generate the forcing term
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)

            # print "-----------------"
            # print "ddy:", self.ddy[d]
            # print "dy:", self.dy[d]
            # print "y:", self.y[d]

            # DMP acceleration
            self.ddy[d] = self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]/tau) + f
            if external_force is not None:
                self.ddy[d] += external_force[d]

            self.ddy[d] *= tau**2

            self.dy[d] += self.ddy[d] * self.dt * cs_args['error_coupling']

            self.y[d] += self.dy[d] * self.dt * cs_args['error_coupling']

            # print "ddy:", self.ddy[d]
            # print "dy:", self.dy[d]
            # print "y:", self.y[d]

        return self.y, self.dy, self.ddy


        # for d in range(self.dmps):

        #     # generate the forcing term
        #     f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)


        #     # print "-----------------"
        #     # print "ddy:", self.ddy[d]
        #     # print "dy:", self.dy[d]
        #     # print "y:", self.y[d]

        #     # DMP acceleration
        #     self.ddy[d] = self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]/tau) + f
        #     if external_force is not None:
        #         self.ddy[d] = external_force[d] * 10

        #     # if contact_judge:
        #     #     self.ddy[d] -= f

        #     self.ddy[d] *= tau**2
        #     self.dy[d] = self.ddy[d] * self.dt * cs_args['error_coupling']

        #     # if external_force is not None:
        #     #     self.dy[d] += external_force[d] * 0.02

        #     self.y[d] += self.dy[d] * self.dt * cs_args['error_coupling']

        #     # print "ddy:", self.ddy[d]
        #     # print "dy:", self.dy[d]
        #     # print "y:", self.y[d]

        # return self.y, self.dy, self.ddy
