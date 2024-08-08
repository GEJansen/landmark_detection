#!/bin/bash

# Parse YAML file and extract parameters
function parse_yaml {
    local prefix=$2
    local s='[[:space:]]*:[[:space:]]*'
    local w='[a-zA-Z0-9_]*'
    local fs=$(echo @|tr @ '\034')
    sed -e "s|^\($w\)$s\(.*\)$|\1$fs\2|" \
        -e "s|^\($w\)$s[\"\']\(.*\)[\"\']$|\1$fs\2|" \
        -e "s|^\($w\)$s\(.*\)$|\1$fs\2|" $1 |
    awk -F$fs '{print $1"=\""$2"\""}'
}

# Source the parsed YAML
eval $(parse_yaml experiment\ settings/second\ step/settings_cephalometric.yaml "config_")

# Run the training script with the extracted parameters
python code/second_step/train_multi_finetune2D.py --data_dir "$config_data_dir" \
                                               --device "$config_device" \
                                               --batch_size "$config_batch_size" \
                                               --learning_rate "$config_learning_rate" \
                                               --max_iters "$config_max_iters" \
                                               --network "$config_network" \
                                               --output_directory "$config_output_directory" \
                                               --rs "$config_rs" \
                                               --voxel_size "$config_voxel_size" \
                                               --log_loss "$config_log_loss" \
                                               --weight_decay "$config_weight_decay" \
                                               --data_augmentation "$config_data_augmentation"
