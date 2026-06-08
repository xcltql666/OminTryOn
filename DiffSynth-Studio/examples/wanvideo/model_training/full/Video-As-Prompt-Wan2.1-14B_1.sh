accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/train \
  --dataset_metadata_path /home/users/astar/cfar/stuchengyou/jcy/xcl/Magic-TryOn/temp/vap_train_data_final/train/metadata.csv \
  --data_file_keys "video,vap_video" \
  --height 832 \
  --width 640 \
  --num_frames 49 \
  --dataset_repeat 7 \
  --model_id_with_origin_paths "xcltql666/Video-As-Prompt-Wan2.1-14B:transformer/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 3 \
  --remove_prefix_in_ckpt "pipe.vap." \
  --output_path "./models/train_0301_new/Video-As-Prompt-Wan2.1-14B_full" \
  --trainable_models "vap" \
  --extra_inputs "vap_video,input_image" \
  --resume_from_checkpoint "/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/models/train_0301_new/Video-As-Prompt-Wan2.1-14B_full/epoch-0.safetensors"
  # --use_gradient_checkpointing_offload






# accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
#   --dataset_base_path test_train \
#   --dataset_metadata_path test_train/metadata_vap.csv \
#   --data_file_keys "video,vap_video" \
#   --height 480 \
#   --width 832 \
#   --num_frames 49 \
#   --dataset_repeat 100 \
#   --model_paths '["/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/checkpoints/Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model-00001-of-00005.safetensors", "/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/checkpoints/Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model-00002-of-00005.safetensors","/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/checkpoints/Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model-00003-of-00005.safetensors","/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/checkpoints/Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model-00004-of-00005.safetensors","/home/users/astar/cfar/stuchengyou/jcy/xcl/DiffSynth-Studio/checkpoints/Video-As-Prompt-Wan2.1-14B/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"]' \
#   --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.vap." \
#   --output_path "./models/train/Video-As-Prompt-Wan2.1-14B_full" \
#   --trainable_models "vap" \
#   --extra_inputs "vap_video,input_image" \
#   --use_gradient_checkpointing_offload