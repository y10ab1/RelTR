# create results folder with date
RESULTS_PATH=../y10ab1_data/results_$(date +%Y-%m-%d_%H-%M-%S)
mkdir $RESULTS_PATH

# create log file with date in results folder
INFERENVE_LOG_PATH=$RESULTS_PATH/inference_log.csv
touch $INFERENVE_LOG_PATH



CUDA_VISIBLE_DEVICES=0 python inference_custom_oi.py --log_path $INFERENVE_LOG_PATH\
                                                    --results_dir $RESULTS_PATH



