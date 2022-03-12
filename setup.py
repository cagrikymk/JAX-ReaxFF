from setuptools import setup, find_packages
import io
import os
import subprocess
import re
import sys
'''
def get_cuda_version():
  try:
    result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE)
    out_str = result.stdout.decode('utf-8')
    regex = r'release (\S+),'
    match = re.search(regex, out_str)
    if match:
      return str(match.group(1))
    else:
      print("nvcc output cannot be parsed to receive the CUDA version")
      return None
  except:
    print("nvcc command cannot be run to find the CUDA version")
    return None


cuda_version = get_cuda_version()
if cuda_version == None:
  print("First CUDA needs to be installed")
  sys.exit(1) # exit on failure

print("Detected cuda version: ", cuda_version)

cuda_version = "cuda{}".format(cuda_version.replace(".",""))
#TODO: Automate installation for cuda dependent jaxlib
'''

INSTALL_REQUIRES = [
  'jax>=0.2.16,<=0.3.1',
  'jaxlib>=0.1.70,<=0.3.0',
  'numba>=0.51.2',
  'numpy>=1.18.0,<1.22.0',
  'scipy>=1.2.1',
  'tabulate>=0.8.9'
]

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()
  
setup(
   name='jaxreaxff',
   version='0.1.0',
   author='Mehmet Cagri Kaymak',
   author_email='kaymakme@msu.edu',
   packages=["jaxreaxff"],
   entry_points={'console_scripts': ['jaxreaxff=jaxreaxff.driver:main',]},
   url='https://github.com/cagrikymk/JAX-ReaxFF',
   license='LICENSE',
   description='A gradient based framework for fast optimization of ReaxFF',
   long_description=long_description,
   long_description_content_type='text/markdown',
   python_requires='>=3.7',
   install_requires=INSTALL_REQUIRES,
   
)
