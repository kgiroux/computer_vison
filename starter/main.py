import config
import cv2
import matplotlib.pyplot as plt

# Loading the configuration from the file
config_loaded = config.load_config()
print("Name: " + config_loaded["name"])
print("Author: " + config_loaded["author"])
print("Configuration version : " + config_loaded["version"])
imgBGR = cv2.imread("../resources/chat.png")
#
plt.subplot(321, label="IMG BGR"), plt.imshow(imgBGR[..., ::-1])
#patchBGR = imgBGR[200:300, 200:400]
patchBGR = imgBGR[0:300, 0:200]

patchHSV = cv2.cvtColor(patchBGR, cv2.COLOR_BGR2HSV)
plt.subplot(322, label="Patch HSV"), plt.imshow(patchHSV[..., ::-1])

color = ('b', 'g', 'r')
hist = cv2.calcHist([patchHSV], channels=[0], mask=None, histSize=[180], ranges=[0, 180])
plt.subplot(323, label="Patch HSV"), plt.plot(hist)

index = 1
for i, col in enumerate(color):
    hist = cv2.calcHist([patchHSV], [i], None,[256],[0,256])
    plt.subplot(323 + index, label="Histogram" + str([i])), plt.plot(hist, color=col)
    plt.xlim([0, 256])
    index +=1
plt.show()
