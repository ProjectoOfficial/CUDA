# CUDA

### Brief
This repository contans multiple examples of code to be run on NVIDIA GPUs, and wants to help you dive deeper into CUDA programming language. CUDA runs in any machine which mounts a NVIDA GPU with compute capability > 3.0 [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus), so make sure your system is supported. You can run CUDA scripts on Linux and Windows machines, but it is mandatory to install NVIDIA drivers and the NVIDIA CUDA Compiler.

### Available code:
- **info**: display CUDA and GPU information
- **sum**: adds two random number -> learn how to move data from CPU to GPU and vice versa and run code on GPU
- **matmul2d**: classical matrix multiplication between two matrices -> learn how to manage multi-dimensional data structure and operate between them

### Prerequisites:
1. install NVIDIA drivers: [https://ubuntu.com/server/docs/nvidia-drivers-installation](https://www.nvidia.com/download/index.aspx)
2. install CUDA on:
   - Ubuntu: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
   - Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
4. check everything with ```nvidia-smi``` and ```nvcc -v```

### Compile
To compile a ```.cu``` file you need to run ```nvcc file_name.cu -o output_file_name```
