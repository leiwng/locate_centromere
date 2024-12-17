import numpy as np

x = [1,3,7,9,13,17,27,9,7,3,1,19,23,37]
print(x)
print(f"len(x)={len(x)}")
# insorted_x = list(np.array(running_median_insort(x, 2)))
insorted_x = list(np.array(x))
print(insorted_x)
print(f"len(insorted_x)={len(insorted_x)}")