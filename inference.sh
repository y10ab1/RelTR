# specify model path
MODEL_PATH='ckpt/checkpoint0149.pth'

# specify image folder
IMAGE_FOLDER='summary_dataset/video/2023-03-26/002DB3028322_video/'

# create results folder with date
RESULTS_PATH=results/results_$(date +%Y-%m-%d_%H-%M-%S)
mkdir $RESULTS_PATH

# create log file with date in results folder
INFERENVE_LOG_PATH=$RESULTS_PATH/inference_log.csv
touch $INFERENVE_LOG_PATH



CUDA_VISIBLE_DEVICES=4 python inference_custom.py --resume $MODEL_PATH\
                                                  --log_path $INFERENVE_LOG_PATH\
                                                    --img_dir $IMAGE_FOLDER\
                                                    # --results_dir $RESULTS_PATH\
