import numpy as np
import quaternion


def quat_mean_eigen(quats):
  """
  Compute the mean of a nxmx4 matrix of quaternions on the first axis.
  Returns an mx4 array.
  """
  n, m, _ = quats.shape

  covs = np.empty((m, 4, 4))

  for i in range(m):
    covs[i, :, :] = quats[:, i, :].T.dot(quats[:, i, :])

  vals, vecs = np.linalg.eigh(covs)

  qm = vecs[:, :, -1]

  qm /= np.linalg.norm(qm, axis=1)[:, np.newaxis]

  return qm


def quat_mean_lsq_one(qs, qm, eps=1e-5):
  """
  Least-square mean of an array of quaternions.
  qs -- (n, 4) array of n quaternions
  qm -- (4,) array of initial state

  Returns a (4,) array, the mean  in the least-square way.
  """
  qs = quaternion.as_quat_array(qs)
  qm = np.quaternion(*qm)
  while True:
    es = qs * qm.inverse()
    es = quaternion.as_rotation_vector(es)

    em = es.mean(axis=0)
    qem = quaternion.from_rotation_vector(em)

    qm = qem * qm

    if np.linalg.norm(em) <= eps:
      break

  return quaternion.as_float_array(qm)[0]


def quat_mean_lsq(quats, qm):
  """
  Least square mean of an array of quaternion sets across the second axis.
  quats -- (n, m, 4) array of quaternions
  qm -- (m, 4) initial mean where to start the iterative process

  Returns -- (m, 4) array, the mean across the second axis
  """
  # import pdb; pdb.set_trace()
  qr = np.zeros_like(qm)
  for k in range(quats.shape[1]):
    qr[k, :] = quat_mean_lsq_one(quats[:, k, :], qm[k, :])

  return qr


def quats_add_residuals(q, rq):
  """
  Add rotation-vector residuals to a quaternion state.
  q -- (m, 4) quaternion state
  rq -- (n, m, 3) rotation matrix vector

  Returns -- (n, m, 4), the quaternions obtained by applying the rotation
  vectors to state q
  """
  q = quaternion.as_quat_array(q)
  rq = quaternion.from_rotation_vector(rq)
  qs = q * rq

  return quaternion.as_float_array(qs)


def quat_add_residuals(q, rq):
  """
  Add rotation-vector residual to quaternion.
  q -- (4,) quaternion
  rq -- (n, 3) residuals

  Return (n, 4) array of quaternions.
  """
  qs = quats_add_residuals(q[np.newaxis, :], rq[:, np.newaxis, :])

  return qs[:, 0, :]


def quat_rotate(q, v, inverse=False):
  """
  Rotate vector v by quaternion q.
  q -- (4,) a quaternion
  v -- (3,) a vector

  Returns (3,) rotated vector
  """
  q = np.quaternion(*q)
  v = np.quaternion(0, *v)

  if inverse:
    q = q.inverse()

  v = q * v * q.inverse()

  return quaternion.as_float_array(v)[0, 1:]


def quat_residuals(qs, q):
  """
  Compute the residuals by 'substracting' q from qs, as rotation vectors.

  qs -- (n, m, 4) an array of (m, 4) quaternion states.
  q -- (m, 4) the quaternion state to substract from qs

  Return:
    (n, m, 3) residuals obtained by qs * q.inv(), as rotation vectors.
  """
  qs = quaternion.as_quat_array(qs)
  q = quaternion.as_quat_array(q)

  iq = np.array([x.inverse() for x in q])

  return quaternion.as_rotation_vector(iq * qs)


def quat_distance(q1, q2):
  """
  A measure of the distance between 2 quaternions.
  """
  return 1 - np.dot(q1, q2) ** 2
