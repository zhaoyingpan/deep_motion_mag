python main.py --phase run --vid_path /n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/bookshelf_50.mp4 --out_dir /home/panzy/deep_mag_test --alpha 50 --exp_name test --device_id 6
python main.py --phase run_temporal --vid_path /n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/baby_20.mp4 --out_dir /home/panzy/deep_mag_test --alpha 20 --fl 0.04 --fh 0.4 --fs 30 --n_filter_tap 2 --filter_type differenceOfIIR --device_id 6
python main.py --phase run_temporal --vid_path /n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/crane_100.avi --out_dir /home/panzy/deep_mag_test --alpha 75 --fl 0.2 --fh 0.25 --fs 24 --n_filter_tap 2 --filter_type fir --device_id 7