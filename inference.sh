# specify model path
MODEL_PATH='ckpt/checkpoint0149.pth'

# specify image folder
IMAGE_FOLDER='summary_dataset/video/2023-03-26/002DB301AA4C_video/'
#IMAGE_PATH='summary_dataset/video/2023-03-26/002DB301AA4C_video/1679789803_1679789805654961_t_video_06.jpg'
#IMAGE_PATH='summary_dataset/image/2023-03-26/002DB301AA4C/1679789803_1679789805654961_t.jpg'

# create results folder with date
RESULTS_PATH=../y10ab1_data/results_$(date +%Y-%m-%d_%H-%M-%S)
mkdir $RESULTS_PATH

# create log file with date in results folder
INFERENVE_LOG_PATH=$RESULTS_PATH/inference_log.csv
touch $INFERENVE_LOG_PATH



CUDA_VISIBLE_DEVICES=7 python inference_custom.py --resume $MODEL_PATH\
                                                  --log_path $INFERENVE_LOG_PATH\
                                                    --results_dir $RESULTS_PATH


# run inference for images in image folder, and break for loop after max iterations
# count=0
# max_iterations=10
# for IMAGE_PATH in $IMAGE_FOLDER*
# do
#     echo 'Inferencing image: '$IMAGE_PATH
#     python inference_custom.py --img_path $IMAGE_PATH\
#                                 --resume $MODEL_PATH\
#                                 --log_path $INFERENVE_LOG_PATH\
#                                 --results_dir $RESULTS_PATH
#     ((count++))
#     if [ $count -eq $max_iterations ]
#     then
#         break
#     fi
# done


