import csv

def convert_log2csv(log_path):
    """extract the content in the logs and convert it to csv"""
    csv_path = log_path.replace('.log', '.csv')

    """ demo of log file, we want the best_rk results
    iteration: 0, best_rk: -1.0000, kd: -1.0000 sp: -1.0000 ps: -1.0000
    iteration: 1, best_rk: -0.6922, kd: -0.6922 sp: -0.8688 ps: -0.7167
    iteration: 2, best_rk: -0.6922, kd: -1.0000 sp: -1.0000 ps: -1.0000
    iteration: 3, best_rk: 0.3466, kd: 0.3466 sp: 0.4693 ps: 0.5405
    iteration: 4, best_rk: 0.5869, kd: 0.5869 sp: 0.7808 ps: 0.7896
    iteration: 5, best_rk: 0.5869, kd: -0.0705 sp: -0.1020 ps: 0.1161
    """

    with open(log_path, 'r') as f:
        lines = f.readlines()
        iter_list = []
        best_rk_list = []

        for line in lines: 
            if not line.startswith('iteration'):
                continue
            else:
                iter_list.append(int(line.split(',')[0].split(':')[1]))
                best_rk_list.append(float(line.split(',')[1].split(':')[1]))
        
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        for i in range(len(iter_list)):
            if iter_list[i] and best_rk_list[i]:  # check if both values are not empty
                writer.writerow([iter_list[i], best_rk_list[i]])


if __name__ == '__main__':
    # log_path = './output/rnd_search_instinct_555_2023_03_23.log'

    log_dir = './output/'
    # traverse all the log files in the directory
    import os
    for file in os.listdir(log_dir):
        if file.endswith('.log'):
            log_path = log_dir + file
            convert_log2csv(log_path)