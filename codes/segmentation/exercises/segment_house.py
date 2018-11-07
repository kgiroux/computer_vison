import numpy as np
import cv2
import matplotlib.pyplot as plt


imBGR = cv2.imread("../ressources/house.jpg")
# Convert to BGR to HSV
imHSV = cv2.cvtColor(imBGR, cv2.COLOR_BGR2HSV)
plt.subplot(341, label="imHSV"), plt.imshow(cv2.cvtColor(imHSV, cv2.COLOR_HSV2BGR)[...,::-1])
# Segment brick
# Select a sub part of the image:
brick_BGR = cv2.imread("../ressources/brick.png")
patch_brick_HSV = cv2.cvtColor(brick_BGR, cv2.COLOR_BGR2HSV)
# Segment Sky
sky_BGR = cv2.imread("../ressources/sky.png")
patch_sky_HSV = cv2.cvtColor(sky_BGR, cv2.COLOR_BGR2HSV)
#Segment Grass
grass_BGR= cv2.imread("../ressources/grass.png")
patch_grass_HSV = cv2.cvtColor(grass_BGR, cv2.COLOR_BGR2HSV)
#Segment Path
pathway_BGR= cv2.imread("../ressources/pathway.png")
patch_pathway_HSV = cv2.cvtColor(pathway_BGR, cv2.COLOR_BGR2HSV)

#Segment Path
column_BGR= cv2.imread("../ressources/white_column.png")
patch_column_HSV = cv2.cvtColor(column_BGR, cv2.COLOR_BGR2HSV)
column_BGR_2= cv2.imread("../ressources/white_column_2.png")
patch_column_2_HSV = cv2.cvtColor(column_BGR_2, cv2.COLOR_BGR2HSV)
column_BGR_3= cv2.imread("../ressources/white_column_3.png")
patch_column_3_HSV = cv2.cvtColor(column_BGR_3, cv2.COLOR_BGR2HSV)
column_BGR_4= cv2.imread("../ressources/white_brick.png")
patch_column_4_HSV = cv2.cvtColor(column_BGR_4, cv2.COLOR_BGR2HSV)
#Segment Path
roof_BGR= cv2.imread("../ressources/roof.png")
patch_roof_HSV = cv2.cvtColor(roof_BGR, cv2.COLOR_BGR2HSV)

#Segment Path
dirt_BGR= cv2.imread("../ressources/dirt.png")
patch_dirt_HSV = cv2.cvtColor(dirt_BGR, cv2.COLOR_BGR2HSV)

#plt.subplot(342,label="Patch Brick"), plt.imshow(cv2.cvtColor(patchBrick, cv2.COLOR_HSV2BGR)[...,::-1])
# Build a histogram:

hist_brick_HS = cv2.calcHist([patch_brick_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_brick_HS /= hist_brick_HS.max()
hist_sky_HS = cv2.calcHist([patch_sky_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_sky_HS /= hist_sky_HS.max()
hist_grass_HS = cv2.calcHist([patch_grass_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_grass_HS /= hist_grass_HS.max()
hist_pathway_HS = cv2.calcHist([patch_pathway_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_pathway_HS /= hist_pathway_HS.max()
hist_dirt_HS = cv2.calcHist([patch_dirt_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_dirt_HS /= hist_dirt_HS.max()
hist_roof_HS = cv2.calcHist([patch_roof_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_roof_HS /= hist_roof_HS.max()
hist_column_HS = cv2.calcHist([patch_column_HSV,patch_column_2_HSV,patch_column_3_HSV,patch_column_4_HSV], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 179, 0, 255])
hist_column_HS /= hist_column_HS.max()



lh_map_brick = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_brick_HS, ranges=[0, 179, 0, 255], scale=255)
lh_map_sky = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_sky_HS, ranges=[0, 179, 0, 255], scale=255)
lh_map_grass = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_grass_HS, ranges=[0, 179, 0, 255], scale=255)
lh_map_pathway = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_pathway_HS, ranges=[0, 179, 0, 255], scale=255)
lh_map_roof = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_roof_HS, ranges=[0, 179, 0, 255], scale=255)
lh_map_column = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_column_HS, ranges=[0, 179, 0, 255], scale=255)
lh_map_dirt = cv2.calcBackProject([imHSV], channels=[0, 1], hist=hist_dirt_HS, ranges=[0, 179, 0, 255], scale=255)


plt.subplot(345, label="calcBackProject Brick"), plt.imshow(lh_map_brick, cmap='gray')
plt.subplot(346, label="calcBackProject Sky"), plt.imshow(lh_map_sky, cmap='gray')
plt.subplot(347, label="calcBackProject Grass"), plt.imshow(lh_map_grass, cmap='gray')
plt.subplot(348, label="calcBackProject Pathway"), plt.imshow(lh_map_pathway, cmap='gray')
plt.figure()
plt.imshow(lh_map_column, cmap='jet')

im_seg = np.zeros((imBGR.shape[0], imBGR.shape[1]), int)
im_seg[lh_map_brick > 0.5 * 255] = 1
im_seg[lh_map_sky > 0.5 * 255] = 2
im_seg[lh_map_grass > 0.5 * 255] = 3
im_seg[lh_map_pathway > 0.5 * 255] = 4
im_seg[lh_map_dirt > 0.5 * 255] = 5
im_seg[lh_map_column > 0.5 * 255] = 6
im_seg[lh_map_roof > 0.5 * 255] = 7


plt.figure()
plt.imshow(im_seg, cmap='jet', vmin=0, vmax=im_seg.max())

plt.show()