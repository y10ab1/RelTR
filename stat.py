import pandas as pd
import matplotlib.pyplot as plt
import argparse


def main(args):
    result_file = args.result_file
    
    # Read the result file
    df = pd.read_csv(result_file)
    print(df.head())
    
    # Create bar chart for subject, relation, object top 30 categories in each subplots and in different colors
    flg, axes = plt.subplots(3, 2, figsize=(15, 15))
    plt.suptitle(f'Statistics for {result_file.split("/")[-2]}', fontsize=20)
    
    # Create bar chart for subject top 30 categories
    df['subject'].value_counts()[:30].plot.bar(ax=axes[0, 0], color='blue', title='Top 30 subject categories')
    df['relation'].value_counts()[:30].plot.bar(ax=axes[1, 0], color='green', title='Top 30 relation categories')
    df['object'].value_counts()[:30].plot.bar(ax=axes[2, 0], color='orange', title='Top 30 object categories')

    

    
    # Create chart for top10 frequent (subject, relation, object) tuples
    df['tuple'] = df['subject'] + ' ' + df['relation'] + ' ' + df['object']
    df['tuple'].value_counts()[:10].plot.bar(ax=axes[0, 1], color='red', title='Top 10 frequent tuples')

    
    # Create line chart for inference time
    df['inference_time'].plot(ax=axes[1, 1], color='black', title='Inference time')
    
    # add average inference time in the upper right corner of last subplot
    axes[1, 1].text(0.5, 0.5, f'Average inference time: {df["inference_time"].mean():.2f} s',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1, 1].transAxes,
                    fontsize=15)
    # add FPS in the upper right corner of last subplot
    axes[1, 1].text(0.5, 0.4, f'FPS: {1/df["inference_time"].mean():.2f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1, 1].transAxes,
                    fontsize=15)
    
    # Hide plot in the last subplot
    axes[2, 1].axis('off')
        
        
    # Set adequate font size for every titles and x-axis, y-axis ticks 
    for ax in axes.flat:
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        
        

    
    
    # rotate x-axis labels
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
     
    # Save the plot
    plt.tight_layout()

    plt.savefig(f'{args.save_dir}/stat_{result_file.split("/")[-2]}.png')
    
    #
    
    
    # Create chart for top10 frequent (subject, relation, object) tuples
    # df['tuple'] = df['subject'] + ' ' + df['relation'] + ' ' + df['object']
    # df['tuple'].value_counts()[:10].plot.bar(color='red', title='Top 10 frequent tuples')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(f'{args.save_dir}/top10_{result_file.split("/")[-2]}.png')
    

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