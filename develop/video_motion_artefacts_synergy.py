import cv2
import numpy as np
from synergy3DMM import SynergyNet

model = SynergyNet()

n_skip = 1

cap = cv2.VideoCapture('/home/bizzego/Downloads/RR.MTS')

framerate = int(cap.get(cv2.CAP_PROP_FPS))

#%%
# n_frames = framerate*100

n=0

results_frame = []

# while(n<10):
while(cap.isOpened()):
    
    i=0
    while (i<n_skip):
        success, img = cap.read()
        i=i+1
    
    if success == False:
        break

    # landmark [[y, x, z], 68 (points)], 
    # mesh [[y, x, z], 53215 (points)], 
    # face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
    
    results_frame.append(model.get_all_outputs(img)[2])    
    n = n+i
    
cap.release()

#%% process results
position_A = []
orientation_A = []
position_B = []
orientation_B = []
for pose in results_frame:
    
    if len(pose) == 2:
        orientation_A.append(pose[0][0])
        position_A.append(pose[0][1])
      
        orientation_B.append(pose[1][0])
        position_B.append(pose[1][1])

position_A = np.array(position_A)
position_B = np.array(position_B)

orientation_A = np.array(orientation_A)
orientation_B = np.array(orientation_B)

orientation = np.stack([orientation_A, orientation_B], axis=2)
position = np.stack([position_A, position_B], axis=2)

import pickle

with open('/home/bizzego/tmp/synergy_out', 'wb') as f:
    pickle.dump({'orientation': orientation, 'position': position}, f)

#%% correct switches
person_A = 0
person_B = 1

prev_pos_A = position[0, :, person_A]
prev_pos_B = position[0, :, person_B]

orientation_out_A = [orientation[0, :, person_A]]
orientation_out_B = [orientation[0, :, person_B]]

position_out_A = [prev_pos_A]
position_out_B = [prev_pos_B]


for i in np.arange(1, len(orientation)):
    
    curr_pos_A = position[i, :, person_A]
    curr_pos_B = position[i, :, person_B]
    
    distance_same = np.linalg.norm(curr_pos_A - prev_pos_A)
    distance_other = np.linalg.norm(curr_pos_B - prev_pos_A)
    
    if distance_same > distance_other:
        #switch
        person_tmp = person_B
        person_B = person_A
        person_A = person_tmp
        
    orientation_out_A.append(orientation[i, :, person_A])
    orientation_out_B.append(orientation[i, :, person_B])
    
    prev_pos_A = position[i, :, person_A]
    prev_pos_B = position[i, :, person_B]
    
    position_out_A.append(prev_pos_A)
    position_out_B.append(prev_pos_B)

orientation_out_A = np.array(orientation_out_A)
orientation_out_B = np.array(orientation_out_B)

position_out_A = np.array(position_out_A)
position_out_B = np.array(position_out_B)

#%%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,2, sharex=True)
axes[0, 0].plot(orientation_out_A)
axes[0, 1].plot(orientation_out_B)

axes[1, 0].plot(position_out_A)
axes[1, 1].plot(position_out_B)





