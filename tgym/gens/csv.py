import csv

import numpy as np
from tgym.core import DataGenerator


class CSVStreamer(DataGenerator):
    """Data generator from csv file.
    The csv file should have headers but no index columns.
    """
    @staticmethod
    def _generator(filename):
        with open(filename, "rb") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                assert len(row) % 2 == 0
                yield np.array(row, dtype=np.float)

    def _iterator_end(self):
        """Return the next element in the generator.
        Rewinds if end of data reached.
        """
        print "End of data reached, rewinding."
        super(self.__class__, self).rewind()

    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        pass
