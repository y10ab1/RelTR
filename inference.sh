# specify model path
MODEL_PATH='ckpt/checkpoint0149.pth'

# specify image folder
IMAGE_FOLDER='summary_dataset/video/2023-03-26/002DB3028322_video/'



CUDA_VISIBLE_DEVICES=4 python inference_custom_vg.py --resume $MODEL_PATH\
                                                    --img_dir $IMAGE_FOLDER\
                                                    -t 0.15\


