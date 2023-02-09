#!/bin/bash

for file in /n/owens-data1/mnt/big2/data/panzy/Total/*.avi;
do
    echo "========================= $file ====================="
    python main.py --phase run --vid_path "$file" --out_dir /n/owens-data1/mnt/big2/data/panzy/deep_motion_mag_videos --alpha 10 --exp_name test --device_id 6
done