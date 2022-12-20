import cv2
# import mediapipe as mp

ratio = 4
n_skip = 1

cap = cv2.VideoCapture('/home/bizzego/Downloads/RR.MTS')

framerate = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('/home/bizzego/Downloads/RR_out.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 
                      framerate/n_skip, (int(cap.get(3)/ratio), int(cap.get(4)/ratio)))

# # Initialize mediapipe pose class.
# mp_pose = mp.solutions.pose

# # Setup the Pose function for videos - for video processing.
# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
#                           min_tracking_confidence=0.7)

# # Initialize mediapipe drawing class - to draw the landmarks points.
# mp_drawing = mp.solutions.drawing_utils

#%%
n_frames = framerate*10

n=0

import numpy as np

def get_centroid(lm):
    x = []
    y = []
    for i in range(10):
        x.append(lm[i].x)
        y.append(lm[i].y)
    x=np.mean(x)
    y=np.mean(y)
    return([x,y])

lm = []
centroids = []

while(cap.isOpened()):
    i=0
    while (i<n_skip):
        success, img = cap.read()
        i=i+1
        
    if success == False:
        break
    
    resultant = pose_video.process(img)
    lm.append(resultant.pose_landmarks)
    
    if resultant.pose_landmarks is not None:
        centroids.append(get_centroid(resultant.pose_landmarks.landmark))

    mp_drawing.draw_landmarks(image=img, landmark_list=resultant.pose_landmarks,
                              connections=mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                           thickness=3, circle_radius=3),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                             thickness=2, circle_radius=2))
    out.write(img[::ratio, ::ratio,:])

    n = n+n_skip
    
cap.release()
out.release()
cv2.destroyAllWindows()


import pickle

with open('/home/bizzego/UniTn/tmp/landmarks', 'wb') as f:
    pickle.dump([lm, centroids], f)

#%%
with open('/home/bizzego/UniTn/tmp/landmarks', 'rb') as f:
    tmp = pickle.load(f)

#%%
centroids = tmp[1]

magnitude = []
for i in np.arange(1, len(centroids)):
    
    x_prev = centroids[i-1][0]
    y_prev = centroids[i-1][1]
    
    x_curr = centroids[i][0]
    y_curr = centroids[i][1]
    
    magnitude.append(np.sqrt( (x_curr - x_prev)**2 +  (y_curr - y_prev)**2 ))
    
    
#%%
magnitude = np.array(magnitude)

import matplotlib.pyplot as plt

framerate_out = framerate/n_skip

t = np.arange(len(magnitude))/framerate_out
plt.plot(t, magnitude)

magnitude_smooth = np.convolve(magnitude, np.ones(5)/5, 'same')
plt.plot(t, magnitude_smooth)


import cv2
from synergy3DMM import SynergyNet

model = SynergyNet()
I = cv2.imread('/home/bizzego/UniTn/software/SynergyNet/img/sample_1.jpg')
# get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
lmk3d, mesh, pose = model.get_all_outputs(I)

#%%
x_ = []
y_ = []
for p in pose:
    y_.append(p[1][0])
    x_.append(p[1][1])
    
import matplotlib.pyplot as plt
plt.plot(y_, x_, 'o')
