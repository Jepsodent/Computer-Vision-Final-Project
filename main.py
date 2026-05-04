import numpy as np
from src.experiment import generate_seq

# X_original , y , ori_groups = generate_seq("original")
# X_dark ,y_dark, ori_dark = generate_seq("dark")
pre_X ,pre_y, pre_groups  = generate_seq("pre")
# np.save("X_original.npy", X_original)
# np.save("y_original.npy", y)
# np.save("ori_groups.npy", ori_groups)
# np.save("X_dark.npy", X_dark)
# np.save("y_dark.npy", y_dark)
# np.save("dark_groups.npy", ori_dark)
np.save("pre_X.npy", pre_X)
np.save("pre_y.npy", pre_y)
np.save("pre_groups.npy", pre_groups)
# np.save("y.npy", y)

#Total Sequences:  64522 # ORI
# Total Sequences:  57848 # DARK

X_dark = np.load("dark_groups.npy")
print(np.load("y_dark.npy").shape)
print(np.unique(X_dark))
