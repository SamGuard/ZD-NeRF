# ZD-NeRF
This is an implementation of a dynamic nerf that adds extra constraints to hopefully improve the performance of dynamic nerfs. This project uses nerfacc for rendering the nerf and its implementation of DNeRF as a base to work on.

## Update
This research project was unsuccessful in creating a better-performing NeRF before the dissertation deadline, and as I no longer have access to large enough GPUs to run this model, I cannot develop it further.

## nerfacc
This project uses the nerfacc module for rendering the NeRF and example code from the repository, which is found here: [https://github.com/KAIR-BAIR/nerfacc](https://github.com/KAIR-BAIR/nerfacc).

## Getting started
To run the code, you must have an environment setup with Cuda enabled, PyTorch, Torchdiffeq and Nerfacc. As well as this, the dnerf-synthetic dataset will need to be obtained and can be found here [https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0). This will then need to be extracted to "/home/ruilongli/data/dnerf/".
Once the environment is set up you can run python3 main.py followed by arguments to start training/rendering.
