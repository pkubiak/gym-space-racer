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



class RocketRacerEnv(gym.Env):
    """
    Spaceship race on track.

    ACTION SPACE:
    0 -> DO NOTHING
    1 -> FIRE LEFT ROCKET
    2 -> FIRE RIGHT ROCKET
    3 -> FIRE MAIN ROCKET
    4 -> OPEN AERODYNAMIC BRAKE

    OBSERVATION SPACE:
    0 -> Velosity
    1 -> Angular velocity
    2..(LIDARS + 1) -> Lidars measurements (distance along lidar to the closest obstacle)
    """

    # Display constants
    VIEWPORT_SIZE = 800
    MAP_SCALE = 0.6
    SPACESHIP_SCALE = 0.03 * MAP_SCALE
    TRACK_COLOR = (1.0, 0.0, 0.0)
    SPACESHIP_COLOR = (0.0, 0.0, 1.0)

    # Spaceship movements constants
    ROTATION_DECAY = 0.995
    VELOCITY_DECAY = 0.999
    ROTATION_ACC = 0.4
    VELOCITY_ACC = 0.4

    # Rewards constants
    LAP_REWARD = 200  # Reward for completing all track stages
    TRACK_STAGES = 20  # Number of stages (each stage is rewarded with LAP_REWARD / TRACK_STAGES)
    CRASH_PENALTY = 100  # Penalty for crashing spaceship
    FINISH_REWARD = 400  # Reward for finishing lap
    TIME_PENALTY = 0.0  # Penalty for each simulation step

    # Spaceship view field constants
    LIDARS = 5
    LIDAR_LENGTH = 40
    VIEW_ANGLE = 90

    # Track constants
    TRACK_SEED = None
    MIN_TRACK_COMPLEXITY = 8
    MAX_TRACK_COMPLEXITY = 24
    TRACK_WIDTH = 0.3

    # Other constants
    FPS = 30
    DEBUG = False  # Display debug lines
    STEPS_LIMIT = 1000

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self.viewer = None

        self.action_space = gym.spaces.Discrete(5)
        assert self.LIDARS % 2 == 1

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.LIDARS + 2,), dtype=np.float32)
        self.spaceship = None

        self.reset()

    def _compute(self):
        shape = []
        ang = self.spaceship.a
        for x, y in self.spaceship.shape:
            x, y = x*np.math.cos(ang) - y * np.math.sin(ang), x * np.math.sin(ang) + y * np.math.cos(ang)
            x = x * self.SPACESHIP_SCALE + self.spaceship.x
            y = y * self.SPACESHIP_SCALE + self.spaceship.y
            shape.append((x, y))
        self.spaceship.points = shape

        self.next_reward = (self.start_position - (self.stage_rewards+1)*2.0*np.pi/self.TRACK_STAGES) % (2.0 * np.pi)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        self.steps += 1

        if action == 0:
            pass
        elif action == 1:
            self.spaceship.va -= self.ROTATION_ACC / self.FPS
        elif action == 2:
            self.spaceship.va += self.ROTATION_ACC / self.FPS
        elif action == 3:
            self.spaceship.vx -= self.VELOCITY_ACC * np.math.sin(self.spaceship.a) / self.FPS
            self.spaceship.vy += self.VELOCITY_ACC * np.math.cos(self.spaceship.a) / self.FPS
        elif action == 4:
            self.spaceship.vx *= 0.91
            self.spaceship.vy *= 0.91

        self.spaceship.x += self.spaceship.vx / self.FPS
        self.spaceship.y += self.spaceship.vy / self.FPS
        self.spaceship.a += self.spaceship.va / self.FPS

        self.spaceship.vx *= self.VELOCITY_DECAY
        self.spaceship.vy *= self.VELOCITY_DECAY
        self.spaceship.va *= self.ROTATION_DECAY

        # -- CHECK SPACESHIPS COLLISIONS --
        self._compute()

        reward, done = 0.0, False
        shape = self.spaceship.points
        body = shape[:4]
        crash = False
        for ei in range(-1, len(body)-1):
            for si in range(len(self.segments)):
                if intersect(body[ei], body[ei+1], self.segments[si][0], self.segments[si][1]):
                    crash = True
                    break

        if crash:
            done = True
            reward -= self.CRASH_PENALTY

        # -- COMPUTE LIDARS --
        state = []
        state.append(np.math.hypot(self.spaceship.vx, self.spaceship.vy))  # Velocity
        state.append(self.spaceship.va)  # Angular velocity

        for i, lidar in enumerate(shape[4:]):
            dist = self.LIDAR_LENGTH
            if not crash:
                for s in self.segments:
                    if intersect(shape[0], lidar, s[0], s[1]):
                        li = intersection(shape[0], lidar, s[0], s[1])
                        d = np.math.hypot(li[0]-shape[0][0], li[1]-shape[0][1])
                        if d < dist:
                            dist = d
            state.append(dist)

        assert len(state) == self.observation_space.shape[0]

        position = np.math.atan2(self.spaceship.y, self.spaceship.x) + np.pi

        if self.last_position > self.next_reward >= position:
            reward += self.LAP_REWARD / self.TRACK_STAGES
            self.stage_rewards += 1

            if self.stage_rewards == self.TRACK_STAGES:
                done = True
                reward += self.FINISH_REWARD

        self.last_position = position

        reward -= self.TIME_PENALTY
        if self.steps >= self.STEPS_LIMIT:
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.map = CircularMap(n=random.randint(self.MIN_TRACK_COMPLEXITY, self.MAX_TRACK_COMPLEXITY), seed=self.TRACK_SEED,  width=self.TRACK_WIDTH)
        self.map.left *= self.MAP_SCALE
        self.map.right *= self.MAP_SCALE
        self.map.cpoints *= self.MAP_SCALE

        # COLLECT SEGEMENTS
        self.segments = []
        for a in (self.map.left, self.map.right):
            for i in range(-1, len(a)-1):
                self.segments.append((a[i], a[i+1]))

        self.spaceship = SimpleNamespace(x=None, y=None, a=0.0, vx=0.0, vy=0.0, va=0.0, color=self.SPACESHIP_COLOR, shape=[(0, 2), (1,-1), (0,0), (-1,-1)], points=None)

        self.spaceship.x = self.map.start.x * self.MAP_SCALE
        self.spaceship.y = self.map.start.y * self.MAP_SCALE
        self.spaceship.a = self.map.start.angle - 0.5*np.pi

        # DEFINE LIDARS
        da = (self.VIEW_ANGLE * (2.0*np.pi) /360) / (self.LIDARS-1)
        for li in range(-(self.LIDARS//2), self.LIDARS//2+1):
            ang = li*da
            self.spaceship.shape.append((-self.LIDAR_LENGTH*np.math.sin(ang), self.LIDAR_LENGTH*np.math.cos(ang) + 2))

        self.start_position = self.last_position = np.math.atan2(self.spaceship.y, self.spaceship.x) + np.pi

        self.stage_rewards = 0  # Number of acquired stage reward
        self.steps = 0

        return self.step(0)[0]

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(self.VIEWPORT_SIZE, self.VIEWPORT_SIZE)
            self.viewer.set_bounds(-1.0, 1.0, -1.0, 1.0)

            # -- DARK BACKGROUND --
            background = rendering.make_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], True)
            background.set_color(0, 0, 0)
            self.viewer.add_geom(background)

        # -- RACE TRACK --
        self.viewer.draw_polyline(self.map.left, color=self.TRACK_COLOR, linewidth=2)
        self.viewer.draw_polyline(self.map.right, color=self.TRACK_COLOR, linewidth=2)
        if self.DEBUG:
            # DRAW CONTROL POINTS
            self.viewer.draw_polyline(self.map.cpoints, color=(0.2, 0.2, 0.0))

        # -- DRAW SPACESHIP --
        p = self.viewer.draw_polyline(self.spaceship.points[:4], color=self.spaceship.color, linewidth=2)
        p.close = True

        # -- DRAW LIDARS
        if self.DEBUG:
            for i in range(self.LIDARS):
                self.viewer.draw_polyline((self.spaceship.points[0], self.spaceship.points[4+i]), color=(0.6, 0.6, 0.6), linewidth=2)

        if self.DEBUG and self.next_reward:
            # DRAW NEXT REWARD LINE
            self.viewer.draw_polyline( [(0, 0), (-2.0*np.math.sin(self.next_reward+0.5*np.pi), 2.0*np.math.cos(self.next_reward+0.5*np.pi))], color=(0, 0, 1), linewidth=1)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
