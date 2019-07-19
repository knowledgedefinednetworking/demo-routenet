# Copyright (c) 2019, Paul Almasan [^1], Sergi Carol [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu

import numpy as np
import os, random
import argparse
import tensorflow as tf
import configparser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Normalization values')
    parser.add_argument("--dir", help="Directories", nargs='+')
    parser.add_argument("--ini", help="Config file")

    args = parser.parse_args()
    limit = 20 # Number of file to sample
    delays = []
    traffic = []
    print("Args dir", args.dir)
    for folder in args.dir:
        f = os.listdir(folder)
        sample = random.sample(f, limit)
        for file in sample:
            print(file)
            record_iterator = tf.python_io.tf_record_iterator(path=folder+'/'+file)
            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                delays += example.features.feature['delay'].float_list.value
                traffic += example.features.feature['traffic'].float_list.value

    print('*' * 10)
    print('Delay')
    print('Mean', round(np.mean(delays),2))
    print('Std', round(np.std(delays), 2))
    print('*' * 10)
    print('Traffic')
    print('Mean', round(np.mean(traffic), 2))
    print('Std', round(np.std(traffic), 2))

    config = configparser.ConfigParser()
    config.optionxform = str  # Disable lowercase conversion
    config.read(args.ini)
    if not os.path.exists(args.ini):
        config['Normalization'] = {'mean_delay': '1', 'std_delay': '1','mean_traffic': '1', 'std_traffic': '1'}
        config.write(open(args.ini, 'w'))

    config['Normalization']['mean_delay'] = str(round(np.mean(delays),2))
    config['Normalization']['std_delay'] = str(round(np.std(delays), 2))
    config['Normalization']['mean_traffic'] = str(round(np.mean(traffic), 2))
    config['Normalization']['std_traffic'] = str(round(np.std(traffic), 2))
    with open(args.ini, 'w') as configfile:
        config.write(configfile)
