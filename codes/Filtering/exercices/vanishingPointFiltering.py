import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import toolbox as toolbox
import particlefilter as pf
#0. Interception
#1. Scoring
#2. Resampling
#3. Motion Update



def newParticles(N):
    particles = None
    # Generate new particles
    return np.random.random((N, 2))*[125., 413.]  # Build an Nx2 array with row being particles, cols the (y,x) coordinate
    # return particles

def motionUpdate(particles):
    # Do the motion update
    particles = particles + np.random.random(size=2) * 15
    return particles

def resamplingFun(particles, weights, resampleN):
    # Modify this function to resample N particles using the weights
    # INFO: particles[i] has weights weights[i]
    newParticles = []

    for i in range(2,len(particles)):
        weights[i] = weights[i-1] + weights[i]

    i = 1
    u = np.zeros(len(particles))
    u[0] = np.random.random()*(1/len(particles))
    for j in range(0,len(particles)):
        #skip the value that are under the threshold
        while i < len(particles) and u[j] > weights[i]:
            i += 1
        newParticles.append([particles[i],1/len(particles)])
        if j+1 < len(particles) :
            u[j+1] = u[j] + 1/len(particles)
    return newParticles

def scoringParticles(z, particles):
    # Compute the scores array that correspond to the particles and the observation (set of candidates vanishing points) z
    #scores = np.zeros_like(len(particles) if particles is not None else 0.) + 1.
    scores = []
    for particle in particles:
        dist_list = []
        for intersection_point in z:
            dist = np.sqrt((intersection_point - particle)**2)
            dist_list.append(dist)
        scores.append(1/np.sum(dist_list))
    return scores


plt.ion()

dataset = '../ressources/kitti/odometry/01/image_2/'

def main():
    hough_thresh = 50

    # Read first frame to extract width, height
    imgName = "%s/%06d.png" % (dataset, 0)
    imBGR = cv2.imread(imgName)
    w, h = imBGR.shape[1], imBGR.shape[0]

    # Initialize the particle filter. (100 particles is enough)
    pf.init(newParticles, 100)

    frameIdx = 500
    while True:
        print("Frame #%d" % frameIdx)

        imgName = "%s/%06d.png" % (dataset, frameIdx)
        imBGR = cv2.imread(imgName)
        if imBGR is None:
            print("File '%s' do not exist. The end ?" % imgName)
            break

        imGRAY = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(imGRAY, 100, 200, apertureSize=3)  # Compute some edges

        plt.figure(1)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.imshow(edges / 255., cmap="gray")

        # Detect lines using left and right criteria
        # Lines are stored as Mx2 matrix. With each row: (rho, theta)
        lines = np.zeros((0, 2))
        linesLeft = cv2.HoughLines(edges, 1, 1*np.pi / 180, hough_thresh, min_theta=-0.9*np.pi/2, max_theta=0)
        linesRight = cv2.HoughLines(edges, 1, 1*np.pi / 180, hough_thresh, min_theta=0, max_theta=0.9*np.pi/2)
        if linesLeft is not None:
            lines = np.concatenate((lines, linesLeft[:, 0][:10]))
        if linesRight is not None:
            lines = np.concatenate((lines, linesRight[:, 0][:10]))
        np.random.shuffle(lines)  # Shuffle the lines to avoid ordering
        # Array that contains the vanishing candidates (intersection between lines)
        vanishingCands = np.zeros((0, 2)) # (y,x)
        # Compute all the vanishing candidates (intersection between two lines)
        if lines is not None:
            print("%d lines detected" % len(lines))
            # Compute intersection between lines (use toolbox.seg_intersect)
            # and fill the "vanishingCands" array (using a np.vstack())
            for rho, theta in lines:
                x1, y1, x2, y2 = toolbox.line_pts(rho, theta)
                plt.plot((x1, x2), (y1, y2), linewidth=1)

                for rho2, theta2 in lines:
                    if rho != rho2 and theta != theta2 :
                        x3, y3, x4, y4 = toolbox.line_pts(rho2, theta2)
                        # Compute interaction between (rho, theta) and (rho2, theta2)
                        inter = toolbox.seg_intersect([y1,x1],[y2,x2],[y3,x3], [y4,x4])
                        # Add the intersection to the vanishing candidates:
                        if inter is not None :
                            vanishingCands = np.vstack([vanishingCands, inter])
                        continue
        else:
            print("No lines detected")

        # Display the vanishing candidates
        plt.scatter(vanishingCands[:, 1], vanishingCands[:, 0], marker='+', s=100)

        plt.xlim(0, imBGR.shape[1])
        plt.ylim(imBGR.shape[0], 0)

        plt.subplot(2, 1, 2)
        plt.imshow(imBGR[..., ::-1] / 255.)

        # Weight the particles using a scoring function you want.
        pf.weighting(vanishingCands, scoringParticles)  # You have to update the scoring function

        # Display stuff
        if pf.particles is not None:
            plt.scatter(pf.particles[:, 1], pf.particles[:, 0], marker='o', s=2 + 200*pf.weights/pf.weights.max(), c='r', edgecolor='none')  # Disply
        plt.xlim(0, imBGR.shape[1])
        plt.ylim(imBGR.shape[0], 0)
        plt.draw()

        # Resample the particles
        pf.resampling(resamplingFun)
        # Apply the motion update
        pf.motionUpdate(motionUpdate)

        # plt.waitforbuttonpress(0.02)
        plt.waitforbuttonpress()
        frameIdx += 1

if __name__ == "__main__":
    main()
    plt.show()