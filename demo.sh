#!/usr/bin/bash

save_path="/yourpath"

# | target_set                      | targets |
# jarvis__megnet                    | e_form | bandgap | 
# jarvis__megnet-bulk               | bulk_modulus
# jarvis__megnet-shear              | shear_modulus
# jarvis__dft_3d_2021               | formation_energy | total_energy | opt_bandgap |
# jarvis__dft_3d_2021-ehull         | ehull |
# jarvis__dft_3d_2021-mbj_bandgap   | mbj_bandgap |

target_set=jarvis__dft_3d_2021
targets=formation_energy
gpu=3
exp_name=DS
reproduciblity_state=3
use_global_stream=True
layer=4 # {4, 7}

CUDA_VISIBLE_DEVICES=${gpu} python demo.py -p latticeformer/default.json \
    --seed 123 \
    --save_path ${save_path} \
    --domain real \
    --num_layers ${layer} \
    --experiment_name ${exp_name}/${targets} \
    --target_set ${target_set} \
    --targets ${targets} \
    --batch_size 256 \
    --reproduciblity_state ${reproduciblity_state} \
    --use_global_stream ${use_global_stream} \
    --pretrained_model /yourpath
