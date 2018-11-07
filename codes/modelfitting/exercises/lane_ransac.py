import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter
plt.ion()


# datax0, datay0 are the dataset x, y
# The line is defined with two points (x1, y1) (x2, y2)
def displayLineModel(datax0, datay0, x1, y1, x2, y2, c):
    plt.figure("Ransac Searching")
    plt.clf()
    # plt.scatter(x0, y0, s=20*np.minimum(1./dist, 1))
    plt.scatter(datax0, datay0, s=1)
    # plt.scatter(datax0[inliersMask], datay0[inliersMask], c=c)
    plt.plot([x1, x2], [y1, y2], 'r', marker='o')
    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)

dataset = '../ressources/kitti/odometry/01/image_2/'
frameIdx = 40
while True:
    print("Frame #%d" % frameIdx)
    imgName = "%s/%06d.png" % (dataset, frameIdx)
    imBGR = cv2.imread(imgName)
    if imBGR is None:
        print("File '%s' does not exist. The end ?" % imgName)
        break
    imGRAY = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(imGRAY, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edges = tophat>50
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.imshow(imBGR[..., ::-1])
    #
    # plt.subplot(2, 1, 2)
    # plt.imshow(edges, cmap="gray")
    # plt.waitforbuttonpress()
    (ey, ex) = np.where(edges > 0)  # Get all non-zero coordinates
    y0 = ey
    x0 = ex

    # Do the ransac,
    Titer = 100
    lines = np.zeros((0, 4))  # Fill that with the lines (x1, y1) (x2, y2) detected
    threshold = 5
    x_best_1, y_best_1, x_best_2, y_best_2 = 0,0,0,0
    m_estimator_sac_best = 0
    x_best_list = []
    y_best_list = []
    best_model_score = 0
    best_inliers_Mask = 0
    list_lines = []
    for i in range(Titer):
        idx = np.random.choice(len(x0), 2, replace=False)  # Picking 2 points random from the edges points
        # Original RANSAC
        # ax + by + c = d
        x1, y1 = x0[idx[0]],y0[idx[0]]
        x2, y2 = x0[idx[1]],y0[idx[1]]
        a = (y1-y2)
        b = (x2-x1)
        c = (x1*y2 - x2*y1)
        dist = np.abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
        #a = (y0[idx[1]] - y0[idx[0]]) / (x0[idx[1]] - x0[idx[0]])
        #b = y0[idx[1]] - a * x0[idx[1]]
        #dist = np.abs((a * x0 - 1 * y0 + b) / (np.sqrt(a * a + b * b)))
        inliersMask = dist <= threshold
        #m_estimator_sac = np.where(dist <= threshold, threshold - dist, 0)
        m_estimator_sac = np.maximum(threshold-dist, 0)
        #modelScore = np.sum(inliersMask)
        modelScore = np.sum(m_estimator_sac)
        if modelScore > best_model_score:
            x_best_1 = x0[idx[0]]
            x_best_2 = x0[idx[1]]
            y_best_1 = y0[idx[0]]
            y_best_2 = y0[idx[1]]
            list_lines.append(([y_best_1, x_best_1, y_best_2, x_best_2], modelScore))
            best_model_score = modelScore
            m_estimator_sac_best = m_estimator_sac
            best_inliers_Mask = inliersMask
        #y1, x1, y2, x2 = 0, 0, 0, 0
        #displayLineModel(x0, y0, x0[idx[0]], y0[idx[0]], x0[idx[1]], y0[idx[1]], 'r')
        #plt.waitforbuttonpress(0.001)

    sorted(list_lines, key=lambda t: t[1], reverse=True)
    if len(list_lines) - 2 > 2:
        list_lines = list_lines[:2]
    lines = np.vstack([lines, [y_best_1, x_best_1, y_best_2, x_best_2]])

    # Display the lines
    plt.figure('Ransac 1')
    plt.clf()
    plt.imshow(imBGR[..., ::-1]//2)
    plt.scatter(x0, y0, color='r', s=1)
    data = np.argwhere(m_estimator_sac_best != 0)
    plt.scatter(x0[best_inliers_Mask], y0[best_inliers_Mask], c='b')
    for i in range(len(list_lines)):
        y1, x1, y2, x2 = list_lines[i][0]
        plt.plot([x1, x2], [y1, y2], 'g', marker='o', linewidth=2)
    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)

    # Use either of the following functions:
    plt.waitforbuttonpress(0.01)  # Wait for a button (click, key) to be pressed
    # frameIdx += 10
    frameIdx += 5
    # plt.waitforbuttonpress()  # Pause for 0.02 second. Useful to get an animation