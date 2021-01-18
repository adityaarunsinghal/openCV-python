import numpy as np
import cv2

img = cv2.imread("fuzzy.png",1)
cv2.imshow("Show", img)

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h = hsv[:, : , 0]
s = hsv[:, : , 1]
v = hsv[:, : , 2]

hsv_split = np.concatenate((h,s,v), axis=1)
# cv2.imshow("hsv", hsv_split)

kernel = np.ones((2,2), 'uint8')

dilate = cv2.dilate(s, kernel, iterations=10)
cv2.imshow("dilated", dilate)

erosion = cv2.erode(dilate, kernel, iterations=10)
cv2.imshow("eroded", erosion)

contours, heirarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
width = 2
index = -1
color = (0, 0, 255)

cv2.drawContours(erosion, contours, index, color, width, )
cv2.imshow("contours", erosion)

# thres_adapt = cv2.adaptiveThreshold(img, 0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# cv2.imshow("adaptive", thres_adapt)

cv2.waitKey(0)
cv2.destroyAllWindows()