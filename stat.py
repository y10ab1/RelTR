import pandas as pd
import matplotlib.pyplot as plt
import argparse


def main(args):
    result_file = args.result_file
    
    # Read the result file
    df = pd.read_csv(result_file)
    print(df.head())
    
    # Create bar chart for subject, relation, object top 30 categories in each subplots and in different colors
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    df['subject'].value_counts()[:30].plot(kind='bar', ax=ax[0], color='red', title='Subject')
    df['relation'].value_counts()[:30].plot(kind='bar', ax=ax[1], color='green', title='Relation')
    df['object'].value_counts()[:30].plot(kind='bar', ax=ax[2], color='blue', title='Object')
    

    
    # Create line chart for inference time
    df['inference_time'].plot(kind='line', ax=ax[3], color='black', title='Inference Time')   
    # add average inference time in the upper right corner of last subplot
    ax[3].text(0.7, 0.9, f'Average inference time: {df["inference_time"].mean():.2f} s', transform=ax[3].transAxes)
    # add FPS in the upper right corner of last subplot
    ax[3].text(0.7, 0.8, f'FPS: {1/df["inference_time"].mean():.2f}', transform=ax[3].transAxes)
    
    
    
    
    # rotate x-axis labels
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
     
    # Save the plot
    fig.tight_layout()

    plt.savefig(f'{args.save_dir}/statistics_{result_file.split("/")[-2]}.png')
    
    # Create chart for top10 frequent (subject, relation, object) tuples
    df['tuple'] = df['subject'] + ' ' + df['relation'] + ' ' + df['object']
    df['tuple'].value_counts()[:10].plot(kind='bar', color='red', title='Top 10 frequent tuples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/top10_{result_file.split("/")[-2]}.png')
    

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_file', type=str, 
                        default='../y10ab1_data/results_2023-05-10_12-12-29/inference_log.csv', 
                        help='Path to the result file')
    
    parser.add_argument('-s', '--save_dir', type=str, default='../y10ab1_data/reltr/statistics',
                        help='Path to the directory to save the statistics')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parse_args()
    main(args)