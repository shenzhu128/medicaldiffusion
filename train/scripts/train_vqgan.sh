# TO TRAIN:
# export PYTHONPATH=$PWD in previous folder
# NCCL_DEBUG=WARN PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py --gpus 1 --default_root_dir /data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints/knee_mri --precision 16 --embedding_dim 256 --n_hiddens 16 --downsample 16 16 16 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 1024 --accumulate_grad_batches 1
# PL_TORCH_DISTRIBUTED_BACKEND=gloo CUDA_VISIBLE_DEVICES=1 python train/train_vqgan.py --gpus 1 --default_root_dir /data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints_generation/knee_mri_gen --precision 16 --embedding_dim 8 --n_hiddens 16 --downsample 8 8 8 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 16384 --accumulate_grad_batches 1
# https://github.com/Lightning-AI/lightning/issues/9641

# PL_TORCH_DISTRIBUTED_BACKEND=gloo CUDA_VISIBLE_DEVICES=1 python train/train_vqgan.py --gpus 1 --default_root_dir /data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints_brats/flair --precision 16 --embedding_dim 8 --n_hiddens 16 --downsample 2 2 2 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 16384 --accumulate_grad_batches 1 --dataset BRATS