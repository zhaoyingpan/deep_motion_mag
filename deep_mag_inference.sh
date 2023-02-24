for fn in /n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/*;

do

    python main.py --phase run --vid_path "$fn" --out_dir /n/owens-data1/mnt/big2/data/panzy/deep_mag_inference --exp_name test --device_id 2

done