import random
from pprint import pprint as pp

import numpy as np
import quaternion

import qukf

class QUKFModel1(qukf.QuaternionUKFModel):
  """
  The internal state:
    additive:
    world_position(3)
    world_velocity(3)
    body_acceleration(1), it's only moving forward
    angular_velocity(3)
    quaternions:
    orientation

  Measurements:
    * angular
    * acceleration
    * position
    * down
    * north
  """
  def f(self, dt, xa, xq):
    w_position = xa[:3]

    w_velocity = xa[3:6]
    b_acceleration = xa[6]
    angular = xa[7:10]

    xq = qukf.quats_add_residuals(xq, angular * dt)

    # world acceleration
    a = qukf.quat_rotate(xq[0], np.array([0, 0, b_acceleration]), inverse=True)

    w_velocity += a * dt
    w_position += w_velocity * dt + 1/2 * dt**2 * a

    return xa, xq

  def h(self, xa, xq, hint=None):
    if hint == 'angular':
      return xa[7:10]
    elif hint == 'acceleration':
      return xa[6:7]
    elif hint == 'position':
      return xa[0:3]
    elif hint == 'down':
      return qukf.quat_rotate(xq[0], np.array([0, 0, -1]))
    elif hint == 'north':
      return qukf.quat_rotate(xq[0], np.array([0, 1, 0]))
    else:
      raise ValueError('Invalid hint {}'.format(hint))

  def q(self, dt):
    return np.diag(
      [5.] * 3 + [2.] * 3 + [1.] + [1.] * 3 + [1.] * 3) * dt

  def r(self, hint=None):
    if hint == 'angular':
      return np.diag([1.] * 3)
    elif hint == 'acceleration':
      return np.diag([1.])
    elif hint == 'position':
      return np.diag([1.] * 3)
    elif hint == 'down':
      return np.diag([1.] * 3)
    elif hint == 'north':
      return np.diag([1.] * 3)
    else:
      raise ValueError('Invalid hint {}'.format(hint))

  def parse_state(self, xa, xq):
    d = {}
    d['position'] = xa[0:3]
    d['velocity'] = xa[3:6]
    d['b_acceleration'] = xa[6:7]
    d['angular'] = xa[7:10]
    d['orientation'] = xq[0]

    return d


class DummyDevice(object):
  """
  It's a device controlled by 3D acceleration and rotation.
  """
  def __init__(self):
    self.w_position = np.zeros(3)
    self.w_velocity = np.zeros(3)
    self.b_acceleration = np.zeros(1)
    self.angular = np.zeros(3)
    self.orientation = np.array([1., 0, 0, 0])

    self.angular_cov = np.diag([1.] * 3)
    self.ba_cov = np.diag([1.])

    self.angular_m_cov = np.diag([1.] * 3)
    self.ba_m_cov = np.diag([1.])
    self.w_position_m_cov = np.diag([1.] * 3)
    self.down_m_cov = np.diag([1.] * 3)
    self.north_m_cov = np.diag([1.] * 3)

    # angular, acceleration, position
    self.measurement_emission_p = [
      ('angular', .7),
      ('acceleration', .5),
      ('down', .5),
      ('north', .5),
      ('position', .1),
    ]

  def move(self, dt):
    """
    Move device, proportionally with dt.
    """
    self.angular += gaussian_noise(self.angular, self.angular_cov * dt)

    self.orientation = qukf.quats_add_residuals(
      self.orientation[np.newaxis, :], self.angular * dt)[0]

    self.b_acceleration += gaussian_noise(
      self.b_acceleration, self.ba_cov * dt)

    acceleration = qukf.quat_rotate(self.orientation,
                                    np.array([0, 0, self.b_acceleration[0]]),
                                    inverse=True)

    self.w_velocity += acceleration * dt
    self.w_position += self.w_velocity * dt + 1/2 * dt**2 * acceleration

  def measure_sensor(self, hint):
    """
    Return one of the 3 possible measurements, according to the multinomial
    distribution.
    """
    if hint == 'angular':
      v = self.angular + gaussian_noise(self.angular, self.angular_m_cov)
    elif hint == 'acceleration':
      v = self.b_acceleration + gaussian_noise(self.b_acceleration,
                                               self.ba_m_cov)
    elif hint == 'position':
      v = self.w_position + gaussian_noise(self.w_position,
                                           self.w_position_m_cov)
    elif hint == 'down':
      v = qukf.quat_rotate(self.orientation, np.array([0, 0, -1.]))
      v += gaussian_noise(v, self.down_m_cov)
    elif hint == 'north':
      v = qukf.quat_rotate(self.orientation, np.array([0, 1., 0]))
      v += gaussian_noise(v, self.north_m_cov)

    return v

  def measure(self):
    result = []
    for h, p in self.measurement_emission_p:
      if random.random() < p:
        result.append((h, self.measure_sensor(h)))

    random.shuffle(result)

    return result

  @property
  def state(self):
    return {
      'w_position': self.w_position,
      'w_velocity': self.w_velocity,
      'b_acceleration': self.b_acceleration,
      'angular': self.angular,
      'orientation': self.orientation,
    }

  def __str__(self):
    return "<DummyDevice: p={}, v={}, ba={}, av={}, o={}".format(
      self.w_position,
      self.w_velocity,
      self.b_acceleration,
      self.angular,
      self.orientation
    )

  __repr__ = __str__


def gaussian_noise(v, cov):
  mean = np.zeros_like(v)
  return np.random.multivariate_normal(mean, cov)


def print_track(track):
  for i in range(len(track)):
    print(track[i].x)


RUN_TIME = 100.     # in seconds
MOVE_INTERVAL = 1


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  device = DummyDevice()
  model = QUKFModel1()

  ukf = qukf.QuaternionUKF(model,
                           xa0=np.zeros(10),
                           xq0=np.array([[1, 0, 0, 0]], dtype=float),
                           p0=np.diag([1.] * 3 + [1.] * 3 + [1.] +\
                                      [1.] * 3 + [1.] * 3))

  t = 0
  true_positions = []
  predicted_positions = []
  measured_positions = []

  while t <= RUN_TIME:
    # import pdb; pdb.set_trace()
    print('At time {}s'.format(t))
    device.move(MOVE_INTERVAL)
    true_positions.append(np.array(device.w_position))

    pp(device.state)

    measurements = device.measure()

    print('Measured', measurements)

    ps = [x for x in measurements if x[0] == 'position']

    if len(ps) > 0:
      measured_positions.append(ps[0][1])

    for i, (h, z) in enumerate(measurements):
      iv = MOVE_INTERVAL if i == 0 else 0
      ukf.predict_and_update(iv, z=z, hint=h)

    print('UKF state')
    d = model.parse_state(*ukf.x)
    pp(d)
    print('Orientation distance', qukf.quat_distance(
      device.orientation, d['orientation']))
    print()


    predicted_positions.append(d['position'])
    t += MOVE_INTERVAL

  # print_track(ukf.track)

  track = ukf.smooth()

  smoothed_positions = np.array(
    [model.parse_state(*t.x_s)['position'] for t in track\
     if t.x_s is not None])
  true_positions = np.array(true_positions)
  predicted_positions = np.array(predicted_positions)
  measured_positions = np.array(measured_positions)

  fig = plt.figure()
  ax = fig.add_subplot('111', projection='3d')

  ax.plot(true_positions[:, 0],
          true_positions[:, 1],
          true_positions[:, 2],
          c='blue')

  ax.plot(predicted_positions[:, 0],
          predicted_positions[:, 1],
          predicted_positions[:, 2],
          c='red')

  ax.scatter(measured_positions[:, 0],
             measured_positions[:, 1],
             measured_positions[:, 2],
             c='lime')

  ax.plot(smoothed_positions[:, 0],
          smoothed_positions[:, 1],
          smoothed_positions[:, 2],
          c='cyan')

  mng = plt.get_current_fig_manager()
  mng.resize(*mng.window.maxsize())

  plt.show()
