from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='htof',
      author='G. Mirek Brandt, Daniel Michalik',
      version='0.2.11',
      python_requires='>=3.5',
      packages=find_packages(),
      package_dir={'htof': 'htof'},
      package_data={'banzai_nres': ['data/hip1_flagged.txt', 'data/hip2_cd_flagged.fits']},
      setup_requires=['pytest-runner'],
      install_requires=requirements,
      tests_require=['pytest>=3.5'])
