# Setup

Installed CUDA (driverless) from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local.
Did first `wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run`.
Then `sh cuda_<version>_linux.run --silent --toolkit --override --installpath=$HOME/cuda`.

## Requirements

Tested with:
- Nvidia drivers: 550.78
- CUDA Version: 12.4 (A100)
- GCC 13.4.0

# Useful Commands

- `nvcc filename.cu -o filename -run`: compile and run;
- `nsys profile --stats=true ./filename`: profile;

- `make`
- `make run`
- `make clean`