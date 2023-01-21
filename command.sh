#!/bin/bash

# Run training
python3 main.py --scene lego --max_steps 5000

# Render
python3 main.py --scene lego --just_render true --model zdnerf_nerf_step28300.pt