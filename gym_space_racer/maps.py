from scipy import interpolate
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from types import SimpleNamespace
from gym_space_racer.geometry import intersect, intersection

class CircularMap:
    """Generate random map in shape of circle"""
    PRECISION = 100

    def __init__(self, n=10, seed=None, width=0.05, debug=False):
        self.n = n
        self.seed = seed or random.randint(0, 10000)
        self.width = width

        np.random.seed(self.seed)

        cp = self._get_control_points(n)

        rand_i = np.random.randint(low=0, high=n, size=(1,))[0]
        dp = cp[(rand_i+1) % len(cp)] - cp[rand_i]

        # print(cp[(rand_i+1) % len(cp)], cp[rand_i], dp)
        self.start = SimpleNamespace(x=cp[rand_i, 0], y=cp[rand_i, 1], angle=math.atan2(dp[1], dp[0]))
        # print(self.start)
        self.cpoints = cp
        if debug:
            plt.plot(cp[:, 0], cp[:, 1], 'x-')

        interp = self._interpolate(cp[:, 0], cp[:, 1], n=self.PRECISION)
        # interp = np.concatenate((interp, [interp[0]]))

        # interp = self._remove_intersections(interp)

        if debug:
            plt.plot(interp[:, 0], interp[:, 1], 'm:')

        left = self._build_track(interp[:, 0], interp[:, 1], 0.5*width)
        if debug:
            plt.plot(left[:, 0], left[:, 1], 'r:')

        self.left = self._remove_intersections(left)

        right = self._build_track(interp[:, 0], interp[:, 1], -0.5*width)
        if debug:
            plt.plot(right[:, 0], right[:, 1], 'g:')
        self.right = self._remove_intersections(right)

    def plot(self):
        plt.plot(self.start[0], self.start[1], 'x')  # start position
        plt.plot(self.right[:, 0], self.right[:, 1], 'g-')
        plt.plot(self.left[:, 0], self.left[:, 1], 'r-')

    def is_valid(self) -> bool:
        for arr in (self.right,):
            for i in range(len(arr)-2):
                if dot(arr[i], arr[i+1], arr[i+1], arr[i+2]) < 0.0:
                    return False
        return True

    def _get_control_points(self, n):
        radius = np.random.normal(1.0, 0.2, size=(n, ))
        radius[-1] = radius[0]
        t = np.linspace(0, 2.0 * np.pi, n+1)[:-1]
        x, y = radius * np.sin(t), radius * np.cos(t)
        res = np.zeros((n+1, 2))
        res[:-1, 0] = x
        res[:-1, 1] = y
        res[-1,0], res[-1, 1] = x[0], y[0]
        return res

    def _interpolate(self, x, y, n):
        tck, u = interpolate.splprep([x,y], s=0.0, k=5, per=True)
        unew = np.linspace(0, 1.0, n)
        out = interpolate.splev(unew, tck)
        res = np.zeros((len(out[0]), 2))
        res[:, 0] = out[0]
        res[:, 1] = out[1]
        return res

    def _build_track(self, x, y, w):
        xs, ys = [], []
        for x0, x1, y0, y1 in zip(x[0:], x[1:], y[0:], y[1:]):
            dx = x1 - x0
            dy = y1 - y0
            d = np.hypot(dx, dy)
            sx, sy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

            px, py = sx + w*dy/d, sy - w*dx/d
            xs.append(px)
            ys.append(py)
        xs.append(xs[0])
        ys.append(ys[0])
        res = np.zeros((len(xs), 2))
        res[:,0]=xs
        res[:,1]=ys
        return res

    def _remove_intersections(self, ps):
        assert (ps[0,0] == ps[-1,0]) and (ps[0,1] == ps[-1,1])

        while True:
            detected = False
            for i in range(0, len(ps)-1):
                for j in range(i+2, len(ps)-2):
                    if intersect(ps[i], ps[i+1], ps[j], ps[j+1]):
                        detected = True
                        ix, iy = intersection(ps[i], ps[i+1], ps[j], ps[j+1])
                        if 2*(j-i) <= len(ps):
                            ps = np.concatenate((ps[:i+1], np.array([[ix, iy]]), ps[j+1:]))
                        else:
                            ps = np.concatenate((np.array([[ix, iy]]), ps[i+1:j+1]))
#                         print(i, j, ps.shape)
                        break
                if detected:
                    break
            if not detected:
                break
        if ps[0,0] != ps[-1,0] or ps[0,1] != ps[-1,1]:
            ps[-1] = ps[0]
        return ps
