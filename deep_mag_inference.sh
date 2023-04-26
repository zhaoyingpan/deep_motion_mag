for fn in /n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/*;

do

    python main.py --phase run --vid_path "$fn" --out_dir /n/owens-data1/mnt/big2/data/panzy/deep_mag_inference --exp_name test --device_id 2

done

python main.py --phase=run_temporal --vid_path /n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/bookshelf_50.mp4 --out_dir /n/owens-data1/mnt/big2/data/panzy/deep_mag_inference --exp_name iir --fl 0.04 --fh 0.4 --fs 30 --n_filter_tap 2 --filter_type differenceOfIIR