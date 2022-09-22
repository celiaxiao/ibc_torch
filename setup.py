from setuptools import setup

install_requires = [
    'scipy',
    'numpy',
    'torch',
    'opencv-python',
    'tqdm',
    'gym',
    'tensorboard',
    'tensorflow',
    'matplotlib',
    'gin',
    'tf_agents',
    'pybullet',
    'tensorflow_datasets',
    'moviepy'

]


setup(
    name='concept',
    version='0.0.1',
    install_requires=install_requires,
    py_modules=['agents', 'ibc', 'losses','network','environments','tests',
        'network.layers','network.utils', 'train','data', 'eval']
)