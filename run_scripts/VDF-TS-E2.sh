CUDA_VISIBLE_DEVICES=0 python src/train_s1.py --arch CA2transformer --data data-bin/how2mcls --train_video_file data/demo_data/pretrain_action.txt --val_video_file data/demo_data/val_action.txt --video_dir data/demo_data/video_action_features --save-dir checkpoints/vdf_s1

CUDA_VISIBLE_DEVICES=0 python src/train_s2_lawd.py --arch LAWDtransformer --data data-bin/how2mcls --train_video_file data/demo_data/train_action.txt --val_video_file data/demo_data/val_action.txt --video_dir data/demo_data/video_action_features --target-dir checkpoints/vdf_s1 --save-dir checkpoints/vdf_s2e2

CUDA_VISIBLE_DEVICES=0 python src/train_s3.py --arch CAX2transformer --data data-bin/how2mcls --train_video_file data/demo_data/train_action.txt --val_video_file data/demo_data/val_action.txt --video_dir data/demo_data/video_action_features --kd_type 0 --save-dir checkpoints/vdf_s3e2 --daencoder-dir checkpoints/vdf_s2e2 --decoder-dir checkpoints/vdf_s1

CUDA_VISIBLE_DEVICES=0 python src/generate.py --data data-bin/how2mcls --video_file data/demo_data/test_action.txt  --video_dir data/demo_data/video_action_features --checkpoint-path checkpoints/vdf_s3e2/checkpoint_last.pt > vdf_ts_e2.out

grep ^H ./vdf_ts_e2.out | cut -f2- | sed -r 's/'$(echo -e "\033")'\[[0-9]{1,2}(;([0-9]{1,2})?)?[mK]//g' >./vdf_ts_e2.txt