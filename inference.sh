# specify model path
MODEL_PATH='ckpt/checkpoint0149.pth'

# specify image folder
IMAGE_FOLDER='summary_dataset/video/2023-03-26/002DB302B7A4_video/'



CUDA_VISIBLE_DEVICES=4 python inference_custom.py --resume $MODEL_PATH\
                                                    --img_dir $IMAGE_FOLDER\


