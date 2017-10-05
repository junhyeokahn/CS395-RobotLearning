import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def getTime(tau, alpha, s):
    """
    Compute time at given s

    Args:
        s: phase variable
        alpha: constant for convergence rate
    Returns:
        t: time
    """
    t = (tau / alpha) * np.log(s)
    return t

def gen_demo(num_traj, tau, noise_scale, cr):
    """
    Trajectory generating function.
    x = (0.6 / tau) * t
    y = 0.15sin( (2pi/0.3)x )
    for tau time duration.

    Args:
        num_traj: The number of trajectory.
                  1 -> w/o noise.
                  2 -> w/ noise.
        tau: time duration.
        noise_scale: scalre multiplied with gaussian noise.
        cr: convergence rate (the value of s at t=tau)

    Returns:
        demo: batch of demonstration
              demonstration1 = ( (t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn) )
              demonstration2 = ( (t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn) )
              demonstration2 includes gaussian noise.
              demo = [demonstration1, demonstration2]
    """
    alpha = np.log(1-cr)
    demo = []
    time_interval = 0.1
    time_stamp = np.arange(0., tau, 0.1)
    x = (0.6/tau) * time_stamp
    y = 0.15*np.sin( (2*np.pi/0.3) * x )
    demo.append(np.vstack((time_stamp, x, y)).T)

    if num_traj == 2:
        noise = np.random.normal(0, 1, (demo[0].shape[0], demo[0].shape[1]-1))/noise_scale
        noise = np.hstack(( np.zeros((demo[0].shape[0], 1)), noise ))
        demo.append(demo[0] + noise)
    return demo

class DMP():
    def __init__(self, cr, tau, kp, kd):
        self.alpha = np.log(1-cr)
        self.tau = tau
        self.kp = kp
        self.kd = kd

    def getPhase(self, t):
        """
        Compute phase variable at a given time

        Args:
            t: time
        Returns:
            s: phase
        """
        s = np.exp( (self.alpha/self.tau) * t )
        return s

    def getPVA(self, traj):
        """
        Compute velocity and acceleration on given trajectory

        Args:
            traj: trajectory

        Returns:
            pva: numpy array of position, velocity, acceleration
                 [(x1,y1,xdot1,ydot1,xddot1,yddot1),
                  ...,
                  (x1,y1,xdot1,ydot1,xddot1,yddot1)]

        """
        pva = np.zeros((traj.shape[0], 6))
        pva[:,0:2] = traj[:,1:]

        for idx in range(traj.shape[0]-1):
            pva[idx, 2:4] = ( pva[idx+1, 0:2] - pva[idx, 0:2] ) \
                            / ( traj[idx+1][0] - traj[idx][0] )
        pva[-1, 2:4] = pva[-2, 2:4]
        for idx in range(traj.shape[0]-1):
            pva[idx, 4:6] = ( pva[idx+1, 2:4] - pva[idx, 2:4] ) \
                            / ( traj[idx+1][0] - traj[idx][0] )
        pva[-1, 4:6] = pva[-2, 4:6]
        return pva

    def learn(self, demo, num_basis, rbf_width, evenly_distributed):
        """
        Learn f(s)
        Use linear interpolation for the case of 1 trajectory
        Use Gaussin basis for the case of 2 trajectories

        Args:
            demo: demonstration batch

        Returns:
        """
        self.num_traj = len(demo)
        if self.num_traj == 1:
            pva = self.getPVA(demo[0])

            start = pva[0][0:2]
            goal = pva[-1][0:2]

            self.f_targ = np.zeros(shape=(demo[0].shape))
            self.f_targ[:, 0] = self.getPhase(demo[0][: ,0])
            self.f_targ[:, 1:] = (self.tau * self.tau * pva[:, 4:6] + self.kd * self.tau * pva[:, 2:4]) / self.kp \
                                - (goal - pva[:, 0:2]) \
                                + (goal - start) * np.vstack((self.f_targ[:, 0], self.f_targ[:, 0])).T

        else:
            batch=[]
            if evenly_distributed:
                self.c = np.arange(0, 1, 1/num_basis)
            else:
                a, b, c, d = 0.01, 0.5, 0.02, 0.99
                self.c = np.arange(a, b, 2*(b-a)/num_basis)
                self.c = np.hstack((self.c, np.arange(c, d, 2*(d-c)/num_basis)))
                num_basis = self.c.shape[0]
            self.h = np.ones(num_basis) * rbf_width
            for i in range(len(demo)):
                pva = self.getPVA(demo[i])
                start = pva[0][0:2]
                goal = pva[-1][0:2]
                f_targ = np.zeros(shape=(demo[i].shape))
                f_targ[:, 0] = self.getPhase(demo[i][: ,0])
                f_targ[:, 1:] = (self.tau * self.tau * pva[:, 4:6] + self.kd * self.tau * pva[:, 2:4]) / self.kp \
                                    - (goal - pva[:, 0:2]) \
                                    + (goal - start) * np.vstack((f_targ[:, 0], f_targ[:, 0])).T
                batch.append(f_targ)


            dataset = np.vstack((batch[0], batch[1]))

            theta = np.zeros(shape=(dataset.shape[0], num_basis))
            for i in range(num_basis):
                theta[:,i] = self.rbf(self.c[i], self.h[i], dataset[:,0])

            left_pseudo = np.matmul(inv(np.matmul(theta.T, theta)), theta.T)
            self.w1 = np.inner(left_pseudo, dataset[:,1] * np.sum(theta, axis=1) / dataset[:,0])
            self.w2 = np.inner(left_pseudo, dataset[:,2] * np.sum(theta, axis=1) / dataset[:,0])

    def rbf(self, c, h, s):
        """
        Guassian basis or Radial Basis Function

        Args:
            c: center
            h: width
            s: phase variable

        Returns:
            f: the value of basis
        """
        f = np.exp(-h*(s-c)*(s-c))
        return f

    def pred(self, s):
        """
        Predict nonlinear function through linear interpolate or function
        approximator

        Args:
            s: phase variable
        Returns:
            f: the value of nonlinear function

        """
        if self.num_traj == 1:
            idx = np.argmin(np.abs(self.f_targ[:, 0] - s))
            if self.f_targ[idx, 0] - s < 0.0001:
                x = self.f_targ[idx, 1]
                y = self.f_targ[idx, 2]
            elif self.f_targ[idx, 0] - s > 0:
                x = np.interp(s, self.f_targ[idx:idx+2, 0], self.f_targ[idx:idx+2, 1])
                y = np.interp(s, self.f_targ[idx:idx+2, 0], self.f_targ[idx:idx+2, 2])
            else:
                x = np.interp(s, self.f_targ[idx-1:idx+1, 0], self.f_targ[idx-1:idx+1, 1])
                y = np.interp(s, self.f_targ[idx-1:idx+1, 0], self.f_targ[idx-1:idx+1, 2])

        else:
            x = np.inner(self.rbf(self.c[:], self.h[:], s), self.w1) * s / np.sum(self.rbf(self.c[:], self.h[:], s))
            y = np.inner(self.rbf(self.c[:], self.h[:], s), self.w2) * s / np.sum(self.rbf(self.c[:], self.h[:], s))

        f = np.array([s, x, y])
        return f

    def plan(self, dt, start, goal, time_dur, obstacle):
        """
        Corresponds to DMP Planning Function

        Args:
            start: [x, y, xdot, ydot, xddot, yddot] @ start
            goal: [x, y, xdot, ydot, xddot, yddot] @ goal

        Returns:
            trajectory: [(t1,x1,y1),(t2,x2,y2), ...,(tn,xn,yn)]
        """
        self.tau = time_dur
        trajectory = np.ndarray(shape=(int(self.tau/dt), 7), dtype=float)
        trajectory[:, 0] = np.arange(0., self.tau, dt)
        trajectory[0, 1:] = start
        for idx, t in enumerate(trajectory[:-1,0]):
            s = self.getPhase(t)
            f = self.pred(s)
            trajectory[idx+1, 3:5] = trajectory[idx, 3:5] + \
                                     trajectory[idx, 5:7] * dt
            trajectory[idx+1, 1:3] = trajectory[idx, 1:3] + \
                                     trajectory[idx, 3:5] * dt
            trajectory[idx+1, 5:7] = (self.kp * (goal[0:2] - trajectory[idx+1, 1:3]) \
                                     - self.kd * self.tau * trajectory[idx+1, 3:5] \
                                     - self.kp * (goal[0:2] - start[0:2]) * s \
                                     + self.kp * f[1:]) / (self.tau^2)
            if obstacle:
                obs = np.array([0.21, 0.])
                # obs = np.array([0.6, 0.]) #obstacle @ goal
                v = trajectory[idx+1, 1:3] - obs
                r = np.sqrt(v[0]*v[0] + v[1]*v[1])
                theta = np.arctan2(v[1], v[0])
                obs_scaler = 50
                obs_width = 100
                trajectory[idx+1, 5:7] += obs_scaler * self.rbf(0, obs_width, r) \
                        * np.array([np.cos(theta), np.sin(theta)])

        return trajectory

def run_dmp(params):
    demo = gen_demo(params.num_traj, params.tau,
                    params.noise_scale, params.convergence_rate)
    dmp = DMP(cr=params.convergence_rate, tau=params.tau,
              kp=np.array([params.kp_x, params.kp_y]),
              kd=np.array([params.kd_x, params.kd_y]))
    dmp.learn(demo, params.num_basis, params.rbf_width, params.evenly_distributed)
    trajectory = dmp.plan(params.dt,
                 np.array([0., 0., 0., 0., 0., 0.]),
                 np.array([params.goal_x, params.goal_y, 0., 0., 0., 0.]),
                 params.tau, params.obstacle)

    plot=True
    if plot:
        if not params.tau == params.time_dur:
            trajectory1 = dmp.plan(params.dt,
                    np.array([0., 0., 0., 0., 0., 0.]),
                    np.array([params.goal_x, params.goal_y, 0., 0., 0., 0.]),
                    params.time_dur, params.obstacle)
            plt.subplot(2,1,1)
            plt.plot(trajectory[:,0], trajectory[:,1],'b',
                     trajectory1[:,0], trajectory1[:,1],'r')
            plt.ylabel('x(m)')
            plt.subplot(2,1,2)
            plt.plot(trajectory[:,0], trajectory[:,2],'b',
                     trajectory1[:,0], trajectory1[:,2],'r')
            plt.ylabel('y(m)')
            plt.xlabel('time(sec)')
            plt.show()

        else:
            if params.obstacle:
                trajectory1 = dmp.plan(params.dt,
                        np.array([0., 0., 0., 0., 0., 0.]),
                        np.array([params.goal_x, params.goal_y, 0., 0., 0., 0.]),
                        params.tau, False)
                plt.plot(trajectory1[:,1], trajectory1[:,2], 'g')

            if params.num_traj == 1:
                plt.plot(demo[0][:,1], demo[0][:,2], 'b',
                         trajectory[:,1], trajectory[:,2], 'r')
                plt.xlabel('x(m)')
                plt.ylabel('y(m)')
                plt.show()
            else:
                plt.plot(demo[0][:,1], demo[0][:,2], 'bo', demo[0][:,1], demo[0][:,2], 'b',
                        demo[1][:,1], demo[1][:,2], 'ro', demo[1][:,1], demo[1][:,2], 'r',
                        trajectory[:,1], trajectory[:,2], 'g', linewidth=3)
                # plt.plot(dmp.c, np.zeros(dmp.c.shape[0]), 'go')
                plt.xlabel('x(m)')
                plt.ylabel('y(m)')
                plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp_x", type=float, default="1000")
    parser.add_argument("--kp_y", type=float, default="1000")
    parser.add_argument("--kd_x", type=float, default="600")
    parser.add_argument("--kd_y", type=float, default="600")
    parser.add_argument("--num_traj", type=int, default="1")
    parser.add_argument("--num_basis", type=int, default="40")
    parser.add_argument("--evenly_distributed", type=bool, default=False)
    parser.add_argument("--rbf_width", type=float, default="1500.")
    parser.add_argument("--tau", type=int, default="20")
    parser.add_argument("--dt", type=float, default="0.001")
    parser.add_argument("--time_dur", type=float, default="20")
    parser.add_argument("--convergence_rate", type=float, default="0.99")
    parser.add_argument("--goal_x", type=float, default="0.6")
    parser.add_argument("--goal_y", type=float, default="0.")
    parser.add_argument("--noise_scale", type=float, default="250")
    parser.add_argument("--obstacle", type=bool, default=False)
    args = parser.parse_args()

    run_dmp(args);

if __name__ == '__main__':
    main()
