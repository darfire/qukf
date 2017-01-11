import numpy as np
import quaternion

from .utils import *


class QuaternionUKFModel(object):
  def f(self, dt, sa, sq):
    """
    Process model. Receives 2-part state, the non-quaternion additive state
    and the quaternion state.
    Returns the updated state as a tuple.
    """
    pass

  def h(self, sa, sq, hint=None):
    """
    Measurement model. Given 2-part state, returns the measurements. We assume
    additive noise for the measurement vector.
    """
    pass

  def q(self, dt):
    """
    Process noise after dt seconds.
    Returns a covariance matrix of dimensionality (n + 3 * nq), where n is
    the size of the additive state and nq is the number of quaternions.
    """
    pass

  def r(self):
    """
    Measurement noise covariance matrix.
    """
    pass


class QuaternionUKF(object):
  """
  An Unscented Kalman Filter implementation that handles unit quaternions.
  """
  def __init__(self, model, xa0, xq0, p0):
    self.x = (np.array(xa0), np.array(xq0))
    self.p = np.array(p0)
    self.model = model

    self.track = []

  def _sigma_transform(self, cov):
    """
    Give an nxn covariance matrix compute the 2n sigma points perturbations.
    Returns a (2nxn) matrix of perturbations.
    """
    n = cov.shape[0]
    try:
      L = np.linalg.cholesky(cov)
    except np.linalg.linalg.LinAlgError:
      raise
    t = np.sqrt(2 * n)

    # 2n x n matrix with the rows as the residuals
    sigmas = np.vstack((t * L.T, -t * L.T))

    return sigmas

  def _add_residual(self, x, res):
    """
    Perturb a single state with a residual. Returns the resulting state.
    """
    xsa, xsq = self._add_residuals(x, res[np.newaxis, :])
    return (xsa[0], xsq[0])

  def _add_residuals(self, x, res):
    """
    Perturb state s with residuals. Res is a matrix of dimensionality
    mx(n+px3).
    Returns m states as a tuple of (mxn, mxpx4) arrays.
    """
    xa, xq = x

    nxa = xa.shape[0]
    n = res.shape[0]

    ra = res[:, :nxa]
    rq = res[:, nxa:].reshape((n, -1, 3))

    xa = xa + ra

    xq = quats_add_residuals(xq, rq)

    return (xa, xq)

  def _compound_mean(self, xs, xm):
    """
    Compute the mean of a number of perturbed states.
    States is an array of (mxn, mxpx4), where m is the number of states,
    n the dimension of the additive state and px4 the dimension of the
    quaternion state.
    Returns the mean state.
    """
    xa, xq = xs

    ma = xa.mean(axis=0)
    mq = quat_mean_lsq(xq, xm[1])

    return (ma, mq)

  def _compound_residuals(self, xs, m):
    """
    Compute xs - m, both for the additive and the quaternion states.
    """
    xa, xq = xs
    ma, mq = m
    n = xa.shape[0]

    # additive state residuals
    ares = xa - ma

    qres = quat_residuals(xq, mq)

    res = np.hstack((ares, qres.reshape((qres.shape[0], -1))))

    return res

  def _compound_residual(self, x2, x1):
    """
    Compute x2 - x1, as above.
    """
    xa, xq = x2
    x2s = (xa[np.newaxis, :], xq[np.newaxis, :])
    res = self._compound_residuals(x2s, x1)

    return res[0]

  def _covariance(self, xs, ys=None):
    """
    Computes the covariance between residuals. If ys is None, compute the
    autocovariance of xs.
    xs an ys need to have the same shape mxn, where the rows are the
    observations and the columns the features.
    Returns the nxn covariance.
    """
    if ys is None:
      ys = xs

    n = xs.shape[0]

    return 1/(2 * n) * xs.T.dot(ys)

  def _copy_state(self, x):
    """
    Duplicate one or more states.
    """
    return tuple(np.array(y) for y in x)

  def _model_f(self, dt, xs):
    """
    Apply the process model on multiple states.
    """
    xs = self._copy_state(xs)
    ys = list(map(lambda x: self.model.f(dt, *x), zip(*xs)))

    res = []
    for i in range(len(ys[0])):
      res.append(np.array([y[i] for y in ys]))

    return res

  def _model_h(self, xs, hint=None):
    """
    Apply the measurement model on multiple states. The hint selects the
    appropriate measurement model.
    """
    xs = self._copy_state(xs)
    zs = list(map(lambda x: self.model.h(*x, hint=hint), zip(*xs)))

    return np.array(zs)

  def predict_and_update(self, dt, z=None, hint=None):
    """
    Do a filtering iteration. If the measurement is missing, leave z as None.
    The hint allows us to work with more than one measurement models, as
    different sensors have different rates of emission (accelerometer vs GPS,
    for example)
    """
    # the predict step
    q = self.model.q(dt)

    ti = TrackItem(q=np.array(q), dt=dt)

    # compute the sigma residuals
    sigma_res = self._sigma_transform(self.p)

    # perturb the mean by the residuals
    x_sigmas = self._add_residuals(self.x, sigma_res)

    # apply process model
    y_sigmas = self._model_f(dt, x_sigmas)

    # compute the new state mean
    y_mean = self._compound_mean(y_sigmas, self.x)

    # compute the new state residuals
    y_res = self._compound_residuals(y_sigmas, y_mean)

    # compute the predict step covariance
    y_cov = self._covariance(y_res) + q

    # end of the predict step

    if z is None:
      # missing measurements, save the current prediction
      self.s = y_mean
      self.p = y_cov
    else:
      # the update step

      # get measurement noise covariance
      r = self.model.r(hint=hint)

      # apply measurement model to y_sigmas
      z_sigmas = list(map(
        lambda y: self.model.h(*y, hint=hint), zip(*y_sigmas)))

      z_sigmas = np.array(z_sigmas)

      z_sigmas = self._model_h(y_sigmas, hint=hint)

      # compute mean and covariance
      z_mean = z_sigmas.mean(axis=0)
      z_res = z_sigmas - z_mean
      z_cov = self._covariance(z_res)

      # compute y/z covariance
      yz_cov = self._covariance(y_res, z_res)

      # compute innovation covariance by adding measurement noise
      v_cov = z_cov + r
      v = z - z_mean

      # kalman gain
      kg = yz_cov.dot(np.linalg.inv(v_cov))

      ti.kg = kg

      # update state and covariance
      self.x = self._add_residual(y_mean, kg.dot(v))
      self.p = y_cov - kg.dot(v_cov).dot(kg.T)

      ti.x = self._clone_state(self.x)
      ti.p = np.array(self.p)

      self.track.append(ti)

  def _clone_state(self, x):
    return tuple(np.array(y) for y in x)

  def smooth(self, track=None):
    if track is None:
      track = self.track

    n = len(track)

    track[n-1].x_s = self._clone_state(track[n-1].x)
    track[n-1].p_s = np.array(track[n-1].p)

    for k in range(n - 2, -1, -1):
      # current record
      tk = track[k]
      # next record
      tkp1 = track[k+1]

      # compute k+1 sigma points based on current filtered state
      sigma_res = self._sigma_transform(tk.p)

      x_sigmas = self._add_residuals(tk.x, sigma_res)

      y_sigmas = self._model_f(tkp1.dt, x_sigmas)

      # mean and covariance
      y_mean = self._compound_mean(y_sigmas, tk.x)

      y_res = self._compound_residuals(y_sigmas, y_mean)

      y_cov = self._covariance(y_res) + tkp1.q

      xy_cov = self._covariance(sigma_res, y_res)

      kg = xy_cov.dot(np.linalg.inv(y_cov))

      v_cov = tkp1.p_s - y_cov

      v = self._compound_residual(tkp1.x_s, y_mean)

      # smoothed state update
      tk.x_s = self._add_residual(tk.x, kg.dot(v))
      tk.p_s = tk.p + kg.dot(v_cov).dot(kg.T)
      tk.kg_s = kg

    return track


class TrackItem(object):
  """
  A record from a filtering run. It saves the states, covariances and
  other interesting data. It can be used in the smoothing stage.
  """
  def __init__(self, **kwargs):
    # filtering data
    # dt, q and kg are the time interval, process covariance and kalman
    # gain that produced this state/covariance combination
    self.x = self.p = self.q = self.kg = self.dt = None
    # smoothed values
    self.x_s = self.p_s = self.kg_s = None

    for k, v in kwargs.items():
      setattr(self, k, v)

  def __str__(self):
    return '<TrackItem: {}>'.format(self.__dict__)

  __repr__ = __str__
