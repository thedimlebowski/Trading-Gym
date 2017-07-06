import csv

import numpy as np
from tgym.gens import CSVStreamer


def test_csv_streamer():
    with open('test.csv', 'w+') as csvfile:
        csv_test = csv.writer(csvfile)
        for i in range(10):
            csv_test.writerow([1] * 10)
    csvstreamer = CSVStreamer(filename='./test.csv')
    for i in range(10):
        assert all(csvstreamer.next() == [1] * 10)
