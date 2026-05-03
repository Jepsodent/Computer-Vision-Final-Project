import numpy as np
from src.experiment import generate_seq

X_original , y = generate_seq("original")
# X_dark , _ = generate_seq("dark")
# X_pre , _ = generate_seq("pre")
np.save("X_original.npy", X_original)
# np.save("X_dark.npy", X_dark)
# np.save("X_pre.npy", X_pre)
np.save("y.npy", y)