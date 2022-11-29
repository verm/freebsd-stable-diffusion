PyTorch and Stable Diffusion on FreeBSD
=======================================

This tutorial was written on FreeBSD 13.1.

##### Table of Contents  
* [Verify and Setup CUDA](#verify-and-setup-cuda)  
* [Initial setup](#initial-setup)
* [Conda](#conda)
* [PyTorch](#pytorch)
* [Stable Diffusion WebUI](#stable-diffusion-webui)

### Notes
For now this only covers those with an [NVIDIA](https://www.nvidia.com/) card 
with CUDA support.

You must have the 
[Linuxulator](https://docs.freebsd.org/en/books/handbook/linuxemu/) installed 
for any of this to work.

This does not use the official [Stable 
Diffusion](https://github.com/CompVis/stable-diffusion) repository but the 
WebUI written by and located at 
[AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

If you wish to install the official repository none of the contents in this 
documentation should change other than checking out the official repository.



Verify and setup CUDA
---------------------

This involves two steps the first is to install `nv-sglrun` in order to check 
for CUDA support which only works for FreeBSD binaries.  The second step is to 
build `uvm_ioctl_override.c` to have the same work for Linux binaries.

This work has been done by **shkhln** you can see his work here 
[libc6-shim](https://github.com/shkhln/libc6-shim).  We will not use this 
directly but the version used for Linux binaries below.

Install **libc6-shim** to get the `nvidia-sglrun` binary.


```
  # pkg install libc6-shim
```

> **Note**
> This requires both **nvidia-driver** and **linux-nvidia-libs**

Check for support using `nvidia-smi`.  Here's a good writeup on what this utility 
displays [Explained Output of Nvidia-smi 
Utility](https://medium.com/analytics-vidhya/explained-output-of-nvidia-smi-utility-fc4fbee3b124).


```
  # nvidia-smi
  Mon Nov 21 15:25:35 2022
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 525.53       Driver Version: 525.53       CUDA Version: N/A      |
  |-------------------------------+----------------------+----------------------+
  <snip>
```


Now use it with `nvidia-sglrun`


```
  # nv-sglrun nvidia-smi
  shim init
  Mon Nov 21 15:25:37 2022       
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 525.53       Driver Version: 525.53       CUDA Version: 12.0     |
  |-------------------------------+----------------------+----------------------+
  <snip>
```

It now says **12.0** for CUDA version instead of **N/A**.  Ignore the driver 
version the one in ports will work just fine.

Also note the **shim init** which gives you an indication 
that the shim is running.

The next part is to build the shim required for Linux binaries located at 
[uvm_ioctl_override.c](https://gist.github.com/shkhln/40ef290463e78fb2b0000c60f4ad797e) 
also written by shkhln.

First install **linux-c7-devtools**.


```
  # pkg install linux-c7-devtools
```


Next get **uvm_ioctl_override.c**.



```
  # fetch https://gist.githubusercontent.com/shkhln/40ef290463e78fb2b0000c60f4ad797e/raw/f640983249607e38af405c95c457ce4afc85c608/uvm_ioctl_override.c
```


Build.


```
  # /compat/linux/bin/cc --sysroot=/compat/linux -m64 -std=c99 -Wall -ldl -fPIC -shared -o dummy-uvm.so uvm_ioctl_override.c
```


This will be tested later as the Linux nvidia-smi works due to the Linux NVIDIA 
libraries.



Initial setup
-------------

Pick a path where you want to install everything.  To keep things easy 
we'll install everything under the same path and use **${PATH}** going forward.

Example.


```
  # PATH="/path/to/somewhere/"
  # mkdir ${PATH}
  # cd ${PATH}
```


 1. When we change a directory assume all the following commands are run 
    under that directory unless we change again.
 2. '#' is a shell login does not reflect being **root**.



Conda
-----

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) from ports using pkg to install Linux Conda.


```
  # pkg install linux-miniconda-installer
  # miniconda-installer
```


  1. Review license and accept license with **yes**.
  1. For a path enter `${PATH}/conda`
     - Example `/path/to/somewhere/conda`
  2. When it asks to run conda init select **no**.
  3. Yes to remove `Miniconda3-latest-Linux-x86_64.sh`.
  4. Choose **got it**.

> **Warning**
> If you hit 'yes' to run `conda init` or run it later it will add a code block to `~/.bashrc`
> that starts with `>>> conda initialize >>>`.


Now let's activate Conda.

First let's activate our environment.  We need to use BASH here most SH scripts 
are written in BASH.  There can be issues running some SH scripts it's not worth 
the trouble.


```
  # bash
  # source ${PATH}/etc/profile.d/conda.sh
  # conda activate
  (base) #
```


The **(base)** lets us know we are in the base environment.



PyTorch
-------
Follow this if you are looking to use PyTorch.  If you want to use Stable 
Diffusion jump to the [next section](#stable-diffusion-webui).


To keep things organised and not have to worry about versions we'll create a 
dedicated environment.  All subsequent commands will be run in this environment.

Create a new environment.  This may give a warning about updating Conda you can 
ignore this and update later if you wish by following the command suggested 
after activating Conda as above.  We will call it `sd` here.  After creation we 
will activate it to install **PyTorch** into it.

> **Note**
> Make sure you hit **y** when it asks you to proceed


```
  (base) # conda create --name pytorch python=3.10
  (base) # conda activate pytorch
```


Let's make sure pip is in the right location.


```
  (pytorch) # which pip
  ${PATH}/conda/envs/pytorch/bin/pip
```


Note the **(pytorch)** in brackets.

This installs the [PyTorch](https://pytorch.org/) distribution with **CUDA**.  


```
  # pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```


Let's see if it works


```
  # python3 -c 'import torch; print(torch.cuda.is_available())'
  ${PATH}/conda/envs/sd/lib/python3.9/site-packages/torch/cuda/__init__.py:83: 
UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:109.)  return torch._C._cuda_getDeviceCount() > 0
  False
```


Now let's try using the shim.



```
  (sd) # LD_PRELOAD="${PATH}/dummy-uvm.so" python3 -c 'import torch; torch.cuda.is_available()'
  True

  (sd) # LD_PRELOAD="${PATH}/dummy-uvm.so" python3 -c 'import torch; print(torch.cuda.get_device_name(0))'
  NVIDIA GeForce RTX 3060
```


You're free to install Stable Diffusion as well it will use its own env.



Stable Diffusion WebUI
----------------------

Next we'll install [Stable 
Diffusion](https://github.com/CompVis/stable-diffusion)  This will use a 
different environment if you installed PyTorch above.  It's better to use 
separate environments so you can avoid version clashing.


> **Note**
> If you installed **PyTorch** above it's safe to run `conda activate` here.



```
  # conda activate
  (base) # git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
```


> **Note**
> Here is where you will switch the repository for 
> `https://github.com/CompVis/stable-diffusion` If you want the original 
> program.  Swap **stable-diffusion-webui** for **stable-diffusion**.  You will 
> also need to use ``environment.yaml`` instead of ``environment-wsl2.yaml`` 
> used below.


```
  (base) # cd stable-diffusion-webui
  (base) # conda env update --file environment-wsl2.yaml --prune
```


You can use this command to update packages when required.  The `--prune` 
command will remove any old packages that aren't used which is very useful for 
debugging.

Let's switch to the **automatic** env now.



```
  (base) # conda activate automatic
  (automatic) # which python3
  ${PATH}/conda/envs/automatic/bin/python3
```


Great.


### Models

Signup and download the [1.4 Model from 
Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)

Download either `sd-v1-4.ckpt` or `sd-v1-4-full-ema.ckpt` depending on which you 
want to use



```
  (automatic) mv sd-v1-4.ckpt ${PATH}/stable-diffusion-webui/models/Stable-diffusion/model.ckpt
```


**model.ckpt** is the default model that will load.  You can have as many as you 
want and load them via the commandline.


Now to install **Git** to allow `launch.py` to install dependencies.


```
  (automatic) # conda install git
```


Next step is to run `launch.py`


```
  (automatic) # LD_PRELOAD=${PATH}/dummy-uvm.so  python3 launch.py
```


You will see some output as it installs required dependencies.


```
Commit hash: 828438b4a190759807f9054932cae3a8b880ddf1
Installing gfpgan
Installing clip
Cloning Stable Diffusion into repositories/stable-diffusion...
Cloning Taming Transformers into repositories/taming-transformers...
Cloning K-diffusion into repositories/k-diffusion...
Cloning CodeFormer into repositories/CodeFormer...
Cloning BLIP into repositories/BLIP...
Installing requirements for CodeFormer
Installing requirements for Web UI
Launching Web UI with arguments: 
${PATH}/conda/envs/automatic/lib/python3.10/site-packages/psutil/_pslinux.py:513: RuntimeWarning: shared, active, inactive memory stats couldn't be determined and were set to 0  warnings.warn(msg, RuntimeWarning)
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
making attention of type 'vanilla' with 512 in_channels
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
making attention of type 'vanilla' with 512 in_channels
Loading weights [7460a6fa] from ${PATH}/stable-diffusion-webui/models/Stable-diffusion/model.ckpt
Global Step: 470000
Applying cross attention optimization (Doggettx).
Model loaded.
Loaded a total of 0 textual inversion embeddings.
Embeddings: 
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.

```


That's it!  You can browse to `http://127.0.0.1:7860` and use the WebUI as you 
wish!

The `launch.py` command takes quite a few arguments you can find by running it 
with `--help`.
