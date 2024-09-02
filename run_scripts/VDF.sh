CUDA_VISIBLE_DEVICES=0 python src/train_base.py --arch CAX2transformer --data data-bin/how2mcls --train_video_file data/demo_data/train_action.txt --val_video_file data/demo_data/val_action.txt --video_dir data/demo_data/video_action_features --save-dir checkpoints/vdf_base

CUDA_VISIBLE_DEVICES=0 python src/generate.py --data data-bin/how2mcls --video_file data/demo_data/test_action.txt  --video_dir data/demo_data/video_action_features --checkpoint-path checkpoints/vdf_base/checkpoint_last.pt > vdf_base.out

grep ^H ./vdf_base.out | cut -f2- | sed -r 's/'$(echo -e "\033")'\[[0-9]{1,2}(;([0-9]{1,2})?)?[mK]//g' >./vdf_base.txt
