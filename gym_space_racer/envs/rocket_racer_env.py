# TODO: COLLISIONS
# TODO: LIDARS values

import gym
from gym import spaces
from gym_space_racer.maps import CircularMap
import random
import numpy as np
from gym.utils import seeding, EzPickle
from types import SimpleNamespace
from gym_space_racer.geometry import intersect, intersection


VIEWPORT_SIZE = 800
MAP_SCALE = 0.6
SPACESHIP_SCALE = 0.03 * MAP_SCALE

ROTATION_DECAY = 0.995
VELOCITY_DECAY = 0.999
ROTATION_ACC = 0.4
VELOCITY_ACC = 0.2

TRACK_STAGES = 20
CRASH_PENALTY = -1000
LAP_REWARD = 200
FINISH_REWARD = 400

TIME_PENALTY = 0.01

FPS = 30

DEBUG = True

STEPS_LIMIT = 1000


LIDARS = 5
LIDAR_LENGTH = 40
VIEW_ANGLE = 90*(2.0*np.pi)/360


class RocketRacerEnv(gym.Env):
    """

    Availabe ACTIONS:
    0 -> DO NOTHING
    1 -> FIRE LEFT ROCKET
    2 -> FIRE RIGHT ROCKET
    3 -> FIRE MAIN ROCKET
    4 -> OPEN AERODYNAMIC BRAKE
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self.viewer = None
        self.seed()

        self.action_space = gym.spaces.Discrete(5)
        assert LIDARS % 2 == 1

        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(LIDARS + 1,), dtype=np.float32)

        self.spaceship = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        self.steps += 1

        if action == 0:
            pass
        elif action == 1:
            self.spaceship.va -= ROTATION_ACC / FPS
        elif action == 2:
            self.spaceship.va += ROTATION_ACC / FPS
        elif action == 3:
            self.spaceship.vx -= VELOCITY_ACC * np.math.sin(self.spaceship.a) / FPS
            self.spaceship.vy += VELOCITY_ACC * np.math.cos(self.spaceship.a) / FPS
        elif action == 4:
            self.spaceship.vx *= 0.91
            self.spaceship.vy *= 0.91

        self.spaceship.x += self.spaceship.vx / FPS
        self.spaceship.y += self.spaceship.vy / FPS
        self.spaceship.a += self.spaceship.va / FPS

        self.spaceship.vx *= VELOCITY_DECAY
        self.spaceship.vy *= VELOCITY_DECAY
        self.spaceship.va *= ROTATION_DECAY

        # -- CHECK SPACESHIPS COLLISIONS --
        shape = []
        ang = self.spaceship.a
        for x, y in self.spaceship.shape:
            x, y = x*np.math.cos(ang) - y *np.math.sin(ang), x * np.math.sin(ang) + y * np.math.cos(ang)
            x = x * SPACESHIP_SCALE + self.spaceship.x
            y = y * SPACESHIP_SCALE + self.spaceship.y
            shape.append((x,y))
        self.spaceship.points = shape

        reward, done = 0.0, False

        body = shape[:4]
        crash = False
        for ei in range(-1, len(body)-1):
            for si in range(len(self.segments)):
                if intersect(body[ei], body[ei+1], self.segments[si][0], self.segments[si][1]):
                    crash = True
                    break

        if crash:
            done = True
            reward -= CRASH_PENALTY

        # -- COMPUTE LIDARS --
        state = []
        for i, lidar in enumerate(shape[4:]):
            dist = np.inf
            if not crash:
                for s in self.segments:
                    if intersect(shape[0], lidar, s[0], s[1]):
                        li = intersection(shape[0], lidar, s[0], s[1])
                        d = np.math.hypot(li[0]-shape[0][0], li[1]-shape[0][1])
                        if d < dist:
                            dist = d
            state.append(dist)
        state.append(np.math.hypot(self.spaceship.vx, self.spaceship.vy))
        # ---------------

        position = np.math.atan2(self.spaceship.y, self.spaceship.x) + np.pi
        next_reward = (self.start_position - (self.stage_rewards+1)*2.0*np.pi/TRACK_STAGES) % (2.0 * np.pi)
        self.next_reward = next_reward

        if self.last_position > next_reward >= position:
            reward += LAP_REWARD / TRACK_STAGES
            self.stage_rewards += 1

            if self.stage_rewards == TRACK_STAGES:
                done = True
                reward += FINISH_REWARD

        self.last_position = position

        reward -= TIME_PENALTY
        if self.steps >= STEPS_LIMIT:
            done = True

        # print(reward, self.stage_rewards, self.spaceship)
        # print(state)
        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.map = CircularMap(n=random.randint(7,25),  width=0.3)
        self.segments = []

        for a in (self.map.left, self.map.right):
            for i in range(-1, len(a)-1):
                self.segments.append((a[i], a[i+1]))

        if self.viewer:
            self.viewer.close()
            self.viewer = None
        # self.viewer = None

        self.map.left *= MAP_SCALE
        self.map.right *= MAP_SCALE
        self.map.cpoints *= MAP_SCALE

        self.spaceship = SimpleNamespace(x=None, y=None, a=0.0, vx=0.0, vy=0.0, va=0.0, color=(1,0,0), shape=[(0, 2), (1,-1), (0,0), (-1,-1)], points=None)

        self.spaceship.x = self.map.start.x * MAP_SCALE
        self.spaceship.y = self.map.start.y * MAP_SCALE
        self.spaceship.a = self.map.start.angle - 0.5*np.pi

        # DEFINE LIDARS
        da = VIEW_ANGLE / (LIDARS-1)
        for li in range(-(LIDARS//2), LIDARS//2+1):
            ang = li*da
            self.spaceship.shape.append((-LIDAR_LENGTH*np.math.sin(ang), LIDAR_LENGTH*np.math.cos(ang) + 2))

        # self.spaceship.vx = 0.1*np.random.normal()
        # self.spaceship.vy = 0.1*np.random.normal()xzz
        # self.spaceship.va = 0.1*np.random.normal()

        self.start_position = self.last_position = np.math.atan2(self.spaceship.y, self.spaceship.x) + np.pi

        self.stage_rewards = 0 # Number of acquired stage reward

        print(self.spaceship, self.last_position)
        self.steps = 0

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.transforms = SimpleNamespace()

            self.viewer = rendering.Viewer(VIEWPORT_SIZE, VIEWPORT_SIZE)
            self.viewer.set_bounds(-1.0, 1.0, -1.0, 1.0)

            # self.objects = SimpleNamespace()
            # -- DARK SPACE --
            background = rendering.make_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], True)
            background.set_color(0,0,0)
            self.viewer.add_geom(background)

            # -- SPACESHIP --
            # spaceship =  rendering.PolyLine([(0, 2),(1, -1), (0, 0), (-1, -1)], True)
            # spaceship.set_linewidth(2)
            self.transforms.spaceship = rendering.Transform(
                translation=(self.spaceship.x, self.spaceship.y),
                rotation=self.spaceship.a,
                scale=(SPACESHIP_SCALE, SPACESHIP_SCALE)
            )
            # spaceship.add_attr(self.transforms.spaceship)
            # spaceship.set_color(*self.spaceship.color)
            # self.viewer.add_geom(spaceship)


            # LIDARS
            # da = self.VIEW_ANGLE / (self.LIDARS-1)
            # for d in range(-(self.LIDARS//2), self.LIDARS//2+1):
            #     ang = d*da
            #     lidar = rendering.Line((0, 2), (-40.0*np.math.sin(ang), 40.0*np.math.cos(ang) + 2))
            #     lidar.add_attr(self.transforms.spaceship)
            #     lidar.set_color(0.6, 0.6, 0.6)
            #     if DEBUG:
            #         self.viewer.add_geom(lidar)

            # -- RACE TRACK --
            left = rendering.make_polyline(self.map.left)
            left.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(left)

            right = rendering.make_polyline(self.map.right)
            right.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(right)

            if DEBUG:
                cpoints = rendering.make_polyline(self.map.cpoints)
                cpoints.set_color(0.2, 0.2, 0.0)
                self.viewer.add_geom(cpoints)

        self.transforms.spaceship.set_translation(self.spaceship.x, self.spaceship.y)
        self.transforms.spaceship.set_rotation(self.spaceship.a)
        # -- DRAW SPACESHIP --
        p = self.viewer.draw_polyline(self.spaceship.points[:4], color=self.spaceship.color, linewidth=2)
        p.close = True

        # -- DRAW LIDARS
        for i in range(LIDARS):
            self.viewer.draw_polyline((self.spaceship.points[0], self.spaceship.points[4+i]), color=(0.6, 0.6, 0.6))

        # self.viewer.draw_polyline( [(0, 0), (self.spaceship.x, self.spaceship.y)], color=(0, 0, 1), linewidth=1)
        if DEBUG and self.next_reward:
            self.viewer.draw_polyline( [(0, 0), (-2.0*np.math.sin(self.next_reward+0.5*np.pi), 2.0*np.math.cos(self.next_reward+0.5*np.pi))], color=(0, 0, 1), linewidth=1)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
