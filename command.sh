#!/bin/bash

# Run training
python3 main.py --scene world_deform_v2 --train_in_order True --max_steps 10000
python3 main.py --scene bouncy --max_steps 100000 --samples 1024 --ray_batch_size 131072 --flow_step 10000 --allow_div False



# Render
python3 main.py --scene bouncy --just_render true --model zdnerf_nerf_step100000.pt --num_renders 51 --allow_div False --render_mode 0
