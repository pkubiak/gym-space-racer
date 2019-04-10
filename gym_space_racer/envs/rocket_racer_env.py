import gym
from gym_space_racer.maps import CircularMap
import random


class RocketRacerEnv(gym.Env):
    SCREEN_SIZE = 600

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.viewer = None

    def step(self, action):
        pass

    def reset(self):
        self.map = CircularMap(n=random.randint(8, 24), width=0.2)
        print(self.map.n, self.map.seed)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.SCREEN_SIZE, self.SCREEN_SIZE)

            background = rendering.make_polygon([(0, 0), (self.SCREEN_SIZE, 0), (self.SCREEN_SIZE, self.SCREEN_SIZE), (0, self.SCREEN_SIZE)], True)
            background.set_color(0,0,0)
            self.viewer.add_geom(background)
        
            spaceship =  rendering.PolyLine([(0, 2),(1, -1), (0, 0), (-1, -1)], True)
            trans = rendering.Transform(translation=(200, 100), scale=(5, 5))

            spaceship.add_attr(trans)
            spaceship.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(spaceship)

            left = rendering.make_polyline(self.map.left)
            left.add_attr(rendering.Transform(scale=(self.SCREEN_SIZE/3, self.SCREEN_SIZE/3), translation=(self.SCREEN_SIZE*0.5, 0.5*self.SCREEN_SIZE)))
            left.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(left)


            right = rendering.make_polyline(self.map.right)
            right.add_attr(rendering.Transform(scale=(self.SCREEN_SIZE/3, self.SCREEN_SIZE/3), translation=(self.SCREEN_SIZE*0.5, 0.5*self.SCREEN_SIZE)))
            right.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(right)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
