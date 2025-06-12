#!/usr/bin/env python3

import numpy as np
import hdf5plot.hdf5_save

a = np.array([[1.0, 12,   20],
              [2,   11,   21],
              [1,   12,   20],
              [2,   11,   21]])
l = hdf5plot.hdf5_save.to_string_array(["x","y","z"])
hdf5plot.hdf5_save.dump("test.hdf5",{"data":a},{"data":l})
print(f"Dumped example at test.hdf5, try opening it with the plotter.")
