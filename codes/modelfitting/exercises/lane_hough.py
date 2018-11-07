import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.ion()


dataset = '../ressources/kitti/odometry/01/image_2/'
frameIdx = 500
while True:
    print("Frame #%d" % frameIdx)

    imgName = "%s/%06d.png" % (dataset, frameIdx)
    imBGR = cv2.imread(imgName)
    if imBGR is None:
        print("File '%s' does not exist. The end ?" % imgName)
        break

    # Detect lines and display them
    # How could you adjust to push the algorithm to detect road lanes ?

    plt.figure(1)
    plt.imshow(imBGR[..., ::-1])
    # Empty figure. Of course it should not be empty.
    plt.draw()

    # Use either of the following functions:
    plt.waitforbuttonpress()  # Wait for a button (click, key) to be pressed
    # plt.waitforbuttonpress()  # Pause for 0.02 second. Useful to get an animation

    frameIdx += 1