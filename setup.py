from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(name='aestetik',
      version='0.1.0',
      description='AESTETIK: AutoEncoder for Spatial Transcriptomics Expression with Topology and Image Knowledge',
      author='KalinNonchev',
      author_email='boo@foo.com',
      license='MIT License',
      long_description_content_type='text/markdown',
      long_description=open('README.md').read(),
      url="https://github.com/ratschlab/aestetik",
      packages=find_packages(where='src'),    # Look in src/
      package_dir={'': 'src'},                # Root is src/
      include_package_data=True,
      # external packages as dependencies,
      install_requires=requirements,
      python_requires='>=3.9'
      )
