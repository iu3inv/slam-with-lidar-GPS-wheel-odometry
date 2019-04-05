import numpy as np
import slam_utils
m=np.array([[1,2,4],[6,7,8],[6,3,9],[6,3,9],[6,3,9]])
print(np.argsort(m.min(axis=1)))
a=slam_utils.solve_cost_matrix_heuristic(m)
print(m)
