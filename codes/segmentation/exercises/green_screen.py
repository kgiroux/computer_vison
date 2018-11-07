import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.ion()

# Open the alpaca video
gs_cap = cv2.VideoCapture('../ressources/green_screen_Alpaca.mov')

# Open the milky way video
bg_cap = cv2.VideoCapture('../ressources/milky_way.mp4')

# If you want to write the video
# ret, gs_frame = gs_cap.read()
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (gs_frame.shape[0], gs_frame.shape[1]))
imgGreenBackBGR = cv2.imread('../ressources/greenscreen.png')
imgGreenBackHSV = cv2.cvtColor(imgGreenBackBGR, cv2.COLOR_HSV2BGR)

hist = cv2.calcHist([imgGreenBackHSV], channels=[0,1], mask=None,  histSize=[180, 256], ranges=[0, 179, 0, 255])
hist /= hist.max()
plt.figure('dist')
while(gs_cap.isOpened()):
    ret, gs_frame = gs_cap.read()  # Capture the green screen image
    ret, bg_frame = bg_cap.read()  # Capture the bg image
    gs_frame = gs_frame / 255.
    bg_frame = bg_frame / 255.

    gs_frame_hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
    lhMap = cv2.calcBackProject([gs_frame_hsv], channels=[0,1], hist=hist, ranges=[0, 179, 0, 255], scale=255)

    # Compute the distance to the green value
    #dist = np.zeros((gs_frame.shape[0], gs_frame.shape[1]))  # Remove that line
    dist = np.sqrt(np.sum((gs_frame - [[[0,255,0]]])**2, axis=2))
    # CODE HERE

    # Replace the
    dist = np.dstack([dist, dist, dist])
    #fx_frame = gs_frame + (1. - dist) *(bg_frame - gs_frame)

    # Plot

    plt.subplot(2, 2, 1)
    plt.imshow(dist, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(gs_frame[..., ::-1])
    plt.subplot(2, 2, 3)
    plt.imshow(bg_frame[..., ::-1])
    plt.subplot(2, 2, 4)
    #plt.imshow(lhMap)
    plt.draw()
    plt.waitforbuttonpress(0.01)

gs_cap.release()
bg_cap.release()
plt.show(block=True)