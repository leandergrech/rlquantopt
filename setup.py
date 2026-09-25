from setuptools import setup, find_packages

setup(
    name='rlquantopt',
    version='2.16.0',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'numpy==1.24.2',
        'pandas==2.1.4',
        'matplotlib',
        'tqdm',
        'torch==2.1.2',
        'PyYAML',
        'qutip==5.0.2',
        'qutip-qip==0.3.1',
        'qutip-qtrl==0.1.1',
        'krotov',
        'gymnasium==0.29.1',
        'stable-baselines3==2.3.1',
        'sb3-contrib==2.3.0',
        'tensorboard',
        'weylchamber'
    ],
)
