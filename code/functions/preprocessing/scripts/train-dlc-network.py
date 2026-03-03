import deeplabcut
import sys
import os

model_path = str(sys.argv[1])

deeplabcut.train_network(config=model_path,
                          max_snapshots_to_keep = 40,
                          maxiters = 1000000,
                          saveiters = 25000,
                         displayiters = 1000)

