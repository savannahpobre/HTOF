from setuptools import setup, find_packages

setup(name='htof',
      author='G. Mirek Brandt, Daniel Michalik',
      version='1.1.0',
      python_requires='>=3.6',
      packages=find_packages(),
      package_dir={'htof': 'htof'},
      package_data={'htof': ['data/*.csv', 'data/*.txt']},
      setup_requires=['pytest-runner'],
      install_requires=['astropy>=2.0', 'pandas>=0.24.0', 'scipy>=1.0.0', 'numpy>=1.16'],
      tests_require=['pytest>=3.5'])
