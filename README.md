# ZD-NeRF
This is an implementation of a dynamic nerf that adds extra constraints to hopefully improve the performance of dynamic nerfs. This project uses nerfacc for the rendering the nerf and its implementation of DNeRF as a base to work off of.

## nerfacc
This project uses the nerfacc module for rendering the NeRF and example code from the repository which is found here: [https://github.com/KAIR-BAIR/nerfacc](https://github.com/KAIR-BAIR/nerfacc).

## Getting started
To run the code, you will need to have an environment setup with cuda enabled, PyTorch, Torchdiffeq and nerfacc. As well as this, data is required, the dnerf-synthetic dataset will need to obtained and can be found here [https://www.dropbox.com/s/25sveotbx2x7wap/logs.zip?dl=0](https://www.dropbox.com/s/25sveotbx2x7wap/logs.zip?dl=0). This will then need to be extracted to "/home/ruilongli/data/dnerf/".
Once the environment is setup you can run python3 main.py followed by arguements to start training/rendering.