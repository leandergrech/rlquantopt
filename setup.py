from setuptools import setup, find_packages

setup(
    name='rlquantopt',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'pandas',
        'matplotlib',
        'qutip',
        'stable-baselines3',
        'gymnasium',
        'scipy',
        'pyyaml',
        'tensorboard'
    ],
)
