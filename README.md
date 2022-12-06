# JAX-ReaxFF
JAX-ReaxFF: A Gradient Based Framework for Extremely Fast Optimization of Reactive Force Fields


### Note: We are currently working on integrating JAX-ReaxFF with JAX-MD. After the integration, the same library will be used for both training and running MD simulation. Once the integration is complete, the code in this repo will be updated to work with JAX-MD.

Existing parameter optimization methods for ReaxFF consist of black-box techniques using genetic algorithms or Monte-Carlo methods. Due to the stochastic behavior of these methods, the optimization process can require millions of error evaluations for complex parameter fitting tasks, significantly hampering the rapid development of high quality parameter sets. 

In this work, we present JAX-ReaxFF, a novel software tool that leverages modern machine learning infrastructure to enable extremely fast optimization of ReaxFF parameters. By calculating gradients of the loss function using the JAX library, we are able to utilize highly effective local optimization methods, such as the limited Broyden–Fletcher–Goldfarb–Shanno (LBFGS) and Sequential Least Squares Programming (SLSQP) methods. 

As a result of the performance portability of JAX, JAX-ReaxFF can execute efficiently on multi-core CPUs, GPUs (or even TPUs). By leveraging the gradient information and modern hardware accelerators, we are able to decrease parameter optimization time for ReaxFF from days to mere minutes. JAX-ReaxFF framework can also serve as a sandbox environment to explore customizing the ReaxFF functional form for more accurate modeling.

Paper: [Jax-ReaxFF](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00363)


## How to Install
Jax-ReaxFF requires JAX and jaxlib ([Jax Repo](https://github.com/google/jax)). <br>
The code is tested with JAX 0.3.0 and jaxlib 0.1.74+.
Since the optimizer is highly more performant on GPUs, GPU version of jaxlib needs to be installed (GPU version supports both CPU and GPU execution). <br>

**1-** Before the installation, a supported version of CUDA and CuDNN are needed (for jaxlib). <br>

**2-** Cloning the Jax-ReaxFF repo:
```
git clone https://github.com/cagrikymk/Jax-ReaxFF
cd Jax-ReaxFF
```

**3-** The installation can be done in a conda environment:
```
conda create -n jax-env python=3.8
conda activate jax-env
```
**4-** Jax-ReaxFF is installed with the following command:
```
pip install .
```
After the setup, Jax-ReaxFF can be accessed via command line interface(CLI) with **jaxreaxff**

To test the installation on a CPU (The JIT compilation time for CPUs drastically higher):
```
jaxreaxff --init_FF Datasets/cobalt/ffield_lit             \
          --params Datasets/cobalt/params                  \
          --geo Datasets/cobalt/geo                        \
          --train_file Datasets/cobalt/trainset.in         \
          --num_e_minim_steps 200                          \
          --e_minim_LR 1e-3                                \
          --out_folder ffields                             \
          --save_opt all                                   \
          --num_trials 1                                   \
          --num_steps 20                                   \
          --init_FF_type fixed                             \
          --backend cpu
```          
**5-** To have the GPU support, jaxlib with CUDA support needs to be installed, otherwise the code can only run on CPUs.
```
# install jaxlib-0.3.0 with Python 3.8, CUDA-11 and cuDNN-8.05 support
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.0+cuda11.cudnn805-cp38-none-manylinux2010_x86_64.whl
```
Other precompilations of jaxlib compatible with different cuda, cudnn and Python versions can be found here: [Jax Releases](https://storage.googleapis.com/jax-releases/jax_releases.html) <br>

To test the GPU version:
```
jaxreaxff --init_FF Datasets/cobalt/ffield_lit             \
          --params Datasets/cobalt/params                  \
          --geo Datasets/cobalt/geo                        \
          --train_file Datasets/cobalt/trainset.in         \
          --num_e_minim_steps 200                          \
          --e_minim_LR 1e-3                                \
          --out_folder ffields                             \
          --save_opt all                                   \
          --num_trials 1                                   \
          --num_steps 20                                   \
          --init_FF_type fixed                             \
          --backend gpu
```   

#### Potential Issues

On a HPC cluster, CUDA might be loaded somewhere different than /usr/local/cuda-xx.x. In this case, XLA compiler might not locate CUDA installation. 
To solve this, we can speficy the cuda directory using XLA_FLAGS:
```
# To see where cuda is installed
which nvcc # will print /opt/software/CUDAcore/11.1.1/bin/nvcc
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_cuda_data_dir=/opt/software/CUDAcore/11.1.1"
```

Another potential issue related XLA compilation on clusters is *RuntimeError: Unknown: no kernel image is available for execution on the device* (potentially related to singularity)
and it can be solved by changing XLA_FLAGS to:

```
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_force_compilation_parallelism=1"
```
This flag can increase the compilation time drastically.


