# create results folder with date
# RESULTS_PATH=../y10ab1_data/results_$(date +%Y-%m-%d_%H-%M-%S)
# mkdir $RESULTS_PATH

# # create log file with date in results folder
# INFERENVE_LOG_PATH=$RESULTS_PATH/inference_log.csv
# touch $INFERENVE_LOG_PATH

IMAGE_FOLDER='summary_dataset/video/2023-03-26/002DB302B7A4_video/'



CUDA_VISIBLE_DEVICES=0 python inference_custom_oi_.py --img_dir $IMAGE_FOLDER\



