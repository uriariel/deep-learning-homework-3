import csv
import itertools

import pandas as pd
import numpy as np

INPUT_DIM = 2
BASE_ORDER = 9


def int_to_binarray(integer: int, base_order: int):
    return [int(c) for c in f'{integer:0{base_order}b}'[::-1]]


def get_data(int_range=1000):
    int_range_product = itertools.product(range(int_range), range(int_range))
    int_range_product_with_sum = [(x, y, x + y) for x, y in int_range_product]
    bin_ints = np.array(
        [[int_to_binarray(x, BASE_ORDER), int_to_binarray(y, BASE_ORDER), int_to_binarray(z, BASE_ORDER)] for
         x, y, z in np.array(int_range_product_with_sum, dtype=np.uint16)])
    return bin_ints


def main():
    bin_ints = get_data(256)
    data = pd.DataFrame(data=bin_ints[:, 2], index=bin_ints[:, 0], columns=bin_ints[:, 1])

    data.to_csv('lstm_adder_data.csv', header=True)
    # with open('adder_data', 'w', newline='') as data_file:
    #     data_writer = csv.writer(data_file, )
    #     for row in get_data(256):
    #         data_writer.writerow(row)


if __name__ == '__main__':
    main()
