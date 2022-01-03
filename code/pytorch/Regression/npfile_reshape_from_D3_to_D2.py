import numpy as np
import pickle
data = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_im_actions.npy')
print(np.shape(data))
DATA = []
for i in range(len(data)):
    for j in range(len(data[i])):
        step = 0
        for k in range(len(data[i][j])):
            # DATA.append(np.hstack([np.array([step]), data[i][j][k]-data[i][j][0]*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])]))  # add step number as label and
            # DATA.append(np.hstack([np.array([step]), data[i][j][k]]))  # add step number as label
            DATA.append(data[i][j][k])
            step += 1

# for j in range(len(data)):
#     step = 0
#     for k in range(len(data[j])):
#         # DATA.append(np.hstack([np.array([step]), data[j][k]-data[j][0]*np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])]))  # add step number as label and
#         # DATA.append(np.hstack([np.array([step]), data[j][k]]))  # add step number as label
#         DATA.append(data[j][k])
#         step += 1

DATA = np.array([DATA])[0]

np.save('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_im_actions_.npy', DATA)
print(np.shape(DATA))
print(DATA[140])