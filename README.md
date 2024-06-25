# JAX-ReaxFF
JAX-ReaxFF: A Gradient Based Framework for Extremely Fast Optimization of Reactive Force Fields

Traditional methods for optimizing ReaxFF parameters rely on slow, stochastic techniques like genetic algorithms or Monte Carlo methods. We present JAX-Reaxff, a new optimizer that leverages modern machine learning infrastructure to dramatically speed up this process.

By utilizing the JAX library to compute gradients of the loss function, we can employ highly efficient local optimization methods, drastically reducing optimization time from days to minutes. JAX-ReaxFF runs efficiently on multi-core CPUs, GPUs, and TPUs, making it versatile and powerful. It also provides a sandbox environment for exploring custom ReaxFF functional forms, enhancing modeling accuracy.

You can learn more about the method in the following papers
(Plase cite them if you utlize this repository):

Original Paper: [Jax-ReaxFF](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00363)

JAX-MD Integration Paper: [End-to-End Differentiable ReaxFF](https://link.springer.com/chapter/10.1007/978-3-031-32041-5_11)

## How to Install
Jax-ReaxFF requires JAX and jaxlib ([Jax Repo](https://github.com/google/jax)). <br>
The code is tested with JAX 0.4.26 - 0.4.30 and jaxlib 0.4.26 - 0.4.30.
Since the optimizer is highly more performant on GPUs, GPU version of jaxlib needs to be installed (GPU version supports both CPU and GPU execution). <br>

**1-** Before the installation, a supported version of CUDA and CuDNN are needed (for jaxlib). Alternatively, one could install the jax-md version that comes with required CUDA libraries. <br>

**2-** Cloning the Jax-ReaxFF repo:
```
git clone https://github.com/cagrikymk/Jax-ReaxFF
cd Jax-ReaxFF
```

**3-** The installation can be done in a conda environment:
```
conda create -n jax-env python=3.9
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
          --init_FF_type fixed                             
```          
**5-** To have the GPU support, jaxlib with CUDA support needs to be installed, otherwise the code can only run on CPUs.
```
pip install -U "jax[cuda12]==0.4.30"
```
You can learn more about JAX installation here: [JAX install guide](https://github.com/google/jax#installation)<br>

After installing the GPU version, the script will automatically utilize the GPU. If the script does not detect the GPU, it will print a warning message.


#### Using Validation Data
```
jaxreaxff --init_FF Datasets/disulfide/ffield_lit             \
          --params Datasets/disulfide/params                  \
          --geo Datasets/disulfide/geo                        \
          --train_file Datasets/disulfide/trainset.in         \
          --use_valid True                                    \
          --valid_file Datasets/disulfide/valSet/trainset.in  \
          --valid_geo_file Datasets/disulfide/valSet/geo      \
          --num_e_minim_steps 200                             \
          --e_minim_LR 1e-3                                   \
          --out_folder ffields                                \
          --save_opt all                                      \
          --num_trials 1                                      \
          --num_steps 20                                      \
          --init_FF_type fixed                             
``` 

#### Potential Issues

On a HPC cluster, CUDA might be loaded somewhere different than /usr/local/cuda-xx.x. In this case, XLA compiler might not locate CUDA installation. This only happens if you install JAX with local CUDA support.
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
