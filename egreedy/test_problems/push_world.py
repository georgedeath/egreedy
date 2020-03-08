"""push_world framework, adapted from [1]_.
See https://github.com/zi-w/Max-value-Entropy-Search for full details.

References
----------
.. [1] Zi Wang and Stefanie Jegelka. 2017.
   Max-value entropy search for efficient Bayesian optimization.
   In Proceedings of the 34th International Conference on Machine Learning.
   PMLR, 3627–3635.
"""
from Box2D import *
from Box2D.b2 import *
import numpy as np
import os

# import pygame without its "Hello from the pygame community" message.
import contextlib
with contextlib.redirect_stdout(None):
    import pygame


# this just makes pygame show what's going on
class guiWorld:
    def __init__(self, fps):
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 750, 750
        self.TARGET_FPS = fps
        self.PPM = 25.0  # pixels per meter
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH,
                                               self.SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('push simulator')
        self.clock = pygame.time.Clock()
        self.screen_origin = b2Vec2(self.SCREEN_WIDTH / (2 * self.PPM),
                                    self.SCREEN_HEIGHT / (self.PPM * 2))
        self.colors = {b2_staticBody: (255, 255, 255, 255),
                       b2_dynamicBody: (163, 209, 224, 255)}

    def draw(self, bodies, bg_color=(64, 64, 64, 0)):
        def my_draw_polygon(polygon, body, fixture):
            vertices = [(self.screen_origin + body.transform * v) * self.PPM
                        for v in polygon.vertices]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            color = self.colors[body.type]

            if isinstance(body.userData, dict):
                color = body.userData['color']

            pygame.draw.polygon(self.screen, color, vertices)

        def my_draw_circle(circle, body, fixture):
            position = ((self.screen_origin + body.transform * circle.pos)
                        * self.PPM)
            position = (position[0], self.SCREEN_HEIGHT - position[1])

            color = self.colors[body.type]

            if isinstance(body.userData, dict):
                color = body.userData['color']

            pygame.draw.circle(self.screen, color, [int(x) for x in position],
                               int(circle.radius * self.PPM))

        b2PolygonShape.draw = my_draw_polygon
        b2CircleShape.draw = my_draw_circle

        # draw the world
        self.screen.fill(bg_color)
        self.clock.tick(self.TARGET_FPS)

        # draw the bodies
        for body in bodies:

            # if we're drawing an object, first draw its target
            if (isinstance(body.userData, dict)
                    and body.userData['type'] == 'object'):

                tx, ty = body.userData['target']
                color = body.userData['color']

                tx = self.PPM * (tx + self.screen_origin[0])
                ty = self.PPM * (-ty + self.screen_origin[1])

                start = (tx - self.PPM / 2, ty - self.PPM / 2)
                end = (tx + self.PPM / 2, ty + self.PPM / 2)
                pygame.draw.line(self.screen, color, start, end, 4)

                start = (tx - self.PPM / 2, ty + self.PPM / 2)
                end = (tx + self.PPM / 2, ty - self.PPM / 2)
                pygame.draw.line(self.screen, color, start, end, 4)

            # now draw its constituent parts
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

        # draw the bounds of the problem, i.e. [-5, -5] to [5, 5]
        points = [(-5, -5), (-5, 5), (5, 5), (5, -5)]
        points = [(self.PPM * (x + self.screen_origin[0]),
                   self.PPM * (y + self.screen_origin[1]))
                  for (x, y) in points]
        pygame.draw.lines(self.screen, (0, 0, 0), True, points, 2)

        pygame.display.flip()


# this is the interface to pybox2d
# edited to allow the target locations to be drawn
class b2WorldInterface:
    def __init__(self, plotting_args=None):
        # default plotting arguments
        self.show = False
        self.save = False
        self.save_every = 10
        self.save_dir = None
        self.save_prefix = 'push'
        
        if plotting_args is not None:
            for arg in ['show', 'save', 'save_every', 
                        'save_dir', 'save_prefix']:
                try:
                    setattr(self, arg, plotting_args[arg])
                except KeyError:
                    pass
                    
            if self.save and not self.show:
                errmsg = 'Due to limitations in pygame, images can only be'
                errmsg += ' saved if they are also shown to screen'
                errmsg += ' therefore if save is True show must also be True'
                raise ValueError(errmsg)
                
            if self.save and self.save_dir is None:
                errmsg = 'Please set a save directory ("save_dir") if'
                errmsg += ' saving images'
                raise ValueError(errmsg)
        
        self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
        self.TARGET_FPS = 100
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.VEL_ITERS, self.POS_ITERS = 10, 10
        self.bodies = []
        self.gui_world = guiWorld(self.TARGET_FPS) if self.show else None

    def add_bodies(self, new_bodies):
        """ add a single b2Body or list of b2Bodies to the world"""
        self.bodies.append(new_bodies)

    def step(self, t=0):
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)
        
        if self.show:
            self.gui_world.draw(self.bodies)

            if self.save and (t % self.save_every == 0):
                save_name = f'{self.save_prefix}_{t:04d}_.png'
                save_path = os.path.join(self.save_dir, save_name)
                
                pygame.image.save(self.gui_world.screen, save_path)


# edited to change the userdata to include color information
class end_effector:
    def __init__(self, b2world_interface, init_pos, base, init_angle,
                 hand_shape='rectangle',
                 hand_size=(0.3, 1),
                 hand_color=(174, 136, 218, 0)):
        world = b2world_interface.world
        self.hand = world.CreateDynamicBody(position=init_pos,
                                            angle=init_angle)
        self.hand_shape = hand_shape
        self.hand_size = hand_size

        # forceunit for circle and rect
        if hand_shape == 'rectangle':
            rshape = b2PolygonShape(box=hand_size)
            self.forceunit = 30.0
        elif hand_shape == 'circle':
            rshape = b2CircleShape(radius=hand_size)
            self.forceunit = 100.0
        elif hand_shape == 'polygon':
            rshape = b2PolygonShape(vertices=hand_size)
        else:
            raise Exception("%s is not a correct shape" % hand_shape)

        self.hand.CreateFixture(shape=rshape, density=0.1, friction=0.1)
        self.hand.userData = {'color': hand_color, 'type': 'hand'}

        friction_joint = world.CreateFrictionJoint(bodyA=base,
                                                   bodyB=self.hand,
                                                   maxForce=2,
                                                   maxTorque=2)
        b2world_interface.add_bodies(self.hand)

    def set_pos(self, pos, angle):
        self.hand.position = pos
        self.hand.angle = angle

    def apply_wrench(self, rlvel=(0, 0), ravel=0):
        avel = self.hand.angularVelocity
        delta_avel = ravel - avel
        torque = self.hand.mass * delta_avel * 30.0
        self.hand.ApplyTorque(torque, wake=True)

        lvel = self.hand.linearVelocity
        delta_lvel = b2Vec2(rlvel) - b2Vec2(lvel)
        force = self.hand.mass * delta_lvel*self.forceunit
        self.hand.ApplyForce(force, self.hand.position, wake=True)

    def get_state(self, verbose=False):
        state = list(self.hand.position) + [self.hand.angle] +  \
                list(self.hand.linearVelocity) + [self.hand.angularVelocity]
        if verbose:
            print_state = ["%.3f" % x for x in state]
            print("position, velocity: (%s), (%s) " %
                  ((", ").join(print_state[:3]), (", ").join(print_state[3:]))
                  )

        return state


# added userdata to set color
def make_1thing(base, b2world_interface, thing_shape, thing_size,
                thing_friction, thing_density, obj_loc,
                obj_color=(123, 128, 120, 0), obj_target=(0, 0)):
    world = b2world_interface.world

    link = world.CreateDynamicBody(position=obj_loc)
    if thing_shape == 'rectangle':
        linkshape = b2PolygonShape(box=thing_size)
    elif thing_shape == 'circle':
        linkshape = b2CircleShape(radius=thing_size)
    elif thing_shape == 'polygon':
        linkshape = b2PolygonShape(vertices=thing_size)
    else:
        raise Exception("%s is not a correct shape" % thing_shape)

    link.CreateFixture(shape=linkshape, density=thing_density,
                       friction=thing_friction)
    friction_joint = world.CreateFrictionJoint(bodyA=base, bodyB=link,
                                               maxForce=5, maxTorque=2)
    link.userData = {'color': obj_color,
                     'type': 'object',
                     'target': obj_target}

    b2world_interface.add_bodies(link)
    return link


# Edited
def make_base(table_width, table_length, b2world_interface):
    world = b2world_interface.world
    base = world.CreateStaticBody(position=(0, 0),
                                  shapes=b2PolygonShape(box=(table_length,
                                                             table_width)))

    b2world_interface.add_bodies(base)
    return base


# Originally called simu_push2 - edited
def simu_push_4D(world, base, thing, robot, xvel, yvel, simulation_steps):
    rvel = b2Vec2(xvel, yvel)

    for t in range(simulation_steps + 100):
        if t < simulation_steps:
            robot.apply_wrench(rvel)

        world.step(t)

    return list(thing.position)


# Originally called simu_push_2robot2thing, adapted for the two
# robot problem, where each robot is represented by a 4D vector of
# its (x, y) position, hand orientation and number of sim steps
def simu_push_8D(world, base, thing, thing2, robot, robot2,
                 xvel, yvel, xvel2, yvel2,
                 simulation_steps, simulation_steps2):

    rvel = b2Vec2(xvel, yvel)
    rvel2 = b2Vec2(xvel2, yvel2)

    for t in range(np.max([simulation_steps, simulation_steps2]) + 100):
        if t < simulation_steps:
            robot.apply_wrench(rvel)

        if t < simulation_steps2:
            robot2.apply_wrench(rvel2)

        world.step(t)

    return (list(thing.position), list(thing2.position))


def push_4D(x, t1_x, t1_y, o1_x, o1_y, plotting_args=None):
    # INPUTS::
    # x = r1_x, r1_y, r1_steps, r1_hand_angle
    # min = [-5; -5; 10; 0];
    # max = [5; 5; 300; 2*pi];
    #
    # Target location: (t1_x, t1_y) in [-5, 5]
    # Object location: (o1_x, o1_y) in [-5, 5]
    r1_x, r1_y, r1_steps, r1_hand_angle = x

    r1_steps = np.round(r1_steps).astype(int)

    # creats the world
    world = b2WorldInterface(plotting_args)

    # object properties and robot properties
    oshape, osize, ofriction, odensity = 'circle', 1, 0.01, 0.05
    robot_shape, robot_size = 'rectangle', (0.3, 1)

    # create the world and place the object in it
    base = make_base(500, 500, world)
    o1 = make_1thing(base, world, oshape, osize, ofriction,
                     odensity, (o1_x, o1_y),
                     obj_color=(0, 0, 150, 0), obj_target=(t1_x, t1_y))

    # robot velocity - always make sure the robot starts by facing the object
    r1_xvel = o1_x - r1_x
    r1_yvel = o1_y - r1_y
    regu = np.linalg.norm([r1_xvel, r1_yvel])
    r1_xvel = (r1_xvel / regu) * 10
    r1_yvel = (r1_yvel / regu) * 10

    # add the robot to the world
    r1 = end_effector(world, (r1_x, r1_y), base, r1_hand_angle,
                      robot_shape, robot_size, hand_color=(0, 0, 255, 0))

    # run simulation and get the object's final location
    o1_loc = simu_push_4D(world, base, o1, r1, r1_xvel, r1_yvel, r1_steps)

    # object's distance to the target location
    d1 = np.linalg.norm(np.array(o1_loc) - [t1_x, t1_y])

    # ensure pygame closes correctly (if open and drawing)
    if draw:
        pygame.display.quit()
        pygame.quit()

    return d1


def push_8D(x, t1_x, t1_y, t2_x, t2_y, o1_x, o1_y, o2_x, o2_y, plotting_args=None):
    # INPUTS::
    # x = rx, ry, steps, hand_angle, rx, ry, steps, hand_angle
    # min = [-5; -5; 10; 0];
    # max = [5; 5; 300; 2*pi];
    # Target locations: (t1_x, t1_y), (t2_x, t2_y)  in [-5, 5]
    # Object locations: (o1_x, o1_y), (o2_x, o2_y) in [-5, 5]
    (r1_x, r1_y, r1_steps, r1_hand_angle,
     r2_x, r2_y, r2_steps, r2_hand_angle) = x

    r1_steps = np.round(r1_steps).astype(int)
    r2_steps = np.round(r2_steps).astype(int)

    # creats the world
    world = b2WorldInterface(plotting_args)

    # object properties and robot properties
    oshape, osize, ofriction, odensity = 'circle', 1, 0.01, 0.05
    robot_shape, robot_size = 'rectangle', (0.3, 1)

    # create the world and place the objects in it
    base = make_base(500, 500, world)
    o1 = make_1thing(base, world, oshape, osize, ofriction,
                     odensity, (o1_x, o1_y), obj_color=(0, 0, 150, 0),
                     obj_target=(t1_x, t1_y))

    o2 = make_1thing(base, world, oshape, osize, ofriction,
                     odensity, (o2_x, o2_y), obj_color=(0, 150, 0, 0),
                     obj_target=(t2_x, t2_y))

    # calculate the velocity of the two robots
    # robot1 velocity - always make sure the robot starts by facing the object
    r1_xvel = o1_x - r1_x
    r1_yvel = o1_y - r1_y
    regu = np.linalg.norm([r1_xvel, r1_yvel])
    r1_xvel = (r1_xvel / regu) * 10
    r1_yvel = (r1_yvel / regu) * 10

    # robot2 velocity - always make sure the robot starts by facing the object
    r2_xvel = o2_x - r2_x
    r2_yvel = o2_y - r2_y
    regu = np.linalg.norm([r2_xvel, r2_yvel])
    r2_xvel = (r2_xvel / regu) * 10
    r2_yvel = (r2_yvel / regu) * 10

    # add both robots to the world
    r1 = end_effector(world, (r1_x, r1_y), base, r1_hand_angle,
                      robot_shape, robot_size, hand_color=(0, 0, 255, 0))
    r2 = end_effector(world, (r2_x, r2_y), base, r2_hand_angle,
                      robot_shape, robot_size, hand_color=(0, 255, 0, 0))

    # run simulation and get the objects' final location
    o1_loc, o2_loc = simu_push_8D(world, base, o1, o2, r1, r2,
                                  r1_xvel, r1_yvel, r2_xvel, r2_yvel,
                                  r1_steps, r2_steps)

    # distances to the target location
    d1 = np.linalg.norm(np.array(o1_loc) - [t1_x, t1_y])
    d2 = np.linalg.norm(np.array(o2_loc) - [t2_x, t2_y])

    # ensure pygame closes correctly (if open and drawing)
    if draw:
        pygame.display.quit()
        pygame.quit()

    return d1 + d2


if __name__ == "__main__":
    # push4 example
    x = np.array([-4, -4, 100, (1 / 4) * np.pi])
    d = push_4D(x, 4, 4, 0, 0, {'show': True})
    print(d)

    # push8 example
    x = np.array([-3, -3, 100, (1 / 4) * np.pi,
                  4, 2.5, 100, (1 / 4) * np.pi])
    d = push_8D(x, 4, 4, 0, -4, -3, 0, 3, 0, {'show': True})
    print(d)
