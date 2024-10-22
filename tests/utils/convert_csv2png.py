import csv 
import os 
from matplotlib import pyplot as plt


if __name__ == '__main__':
    csv_dir = './output/'
    # traverse all the csv files in the directory
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            csv_path = csv_dir + file

            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                iter_list = []
                best_rk_list = []
                for row in reader:
                    iter_list.append(int(row[0]))
                    best_rk_list.append(float(row[1]))

                plt.plot(iter_list, best_rk_list, label=file.split('.')[0])


        plt.ylim(0, 0.85)
        plt.legend()
        plt.ylabel('best_rk')
        plt.xlabel('iteration')
        plt.title('best_rk vs. iteration')
        plt.savefig('./output/best_rk_vs_iteration.png')

            