import torch
import pandas as pd
import os
import PIL

import os
# assign directory
directory = 'data'

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
    else:
        for filename2 in os.listdir(f):
            f2 = os.path.join(f, filename2)
            if os.path.isfile(f2):
                print(f2)
            else:
                for filename3 in os.listdir(f2):
                    f3 = os.path.join(f2, filename3)
                    if os.path.isfile(f3):
                        print(f3)
