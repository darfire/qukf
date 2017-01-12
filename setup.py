from setuptools import setup

setup(
  name='qukf',
  version='0.0.1',
  description='Quaternion-aware Unscented Kalman Filter',
  license='MIT',
  author='Doru Arfire',
  url='https://github.com/darfire/qukf',
  packages=['qukf'],
  install_requires=[
    'numpy-quaternion',
  ],
)
