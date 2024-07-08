from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['utils', 'lib', 'cfgs'],
    package_dir={'': 'src'}
)

setup(**d)
