from pathlib import Path
from matplotlib import pyplot as plt
import cv2 
import os
import numpy as np
import sys

ROOT_PATH = Path.cwd() / Path("Images")

def show_img(img):
  plt.imshow(img)
  plt.title('my picture')
  plt.show()

# link - https://learnopencv.com/contour-detection-using-opencv-python-c/
def use_image_single_channel(image):
    # B, G, R channel splitting
    blue, green, red = cv2.split(image)
    
    # detect contours using blue channel and without thresholding
    contours1, hierarchy1 = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    # draw contours on the original image
    image_contour_blue = image.copy()
    cv2.drawContours(image=image_contour_blue, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # see the results
    cv2.imshow('Contour detection using blue channels only', image_contour_blue)
    cv2.waitKey(0)
    cv2.imwrite('blue_channel.jpg', image_contour_blue)
    cv2.destroyAllWindows()
    
    # detect contours using green channel and without thresholding
    contours2, hierarchy2 = cv2.findContours(image=green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_contour_green = image.copy()
    cv2.drawContours(image=image_contour_green, contours=contours2, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # see the results
    cv2.imshow('Contour detection using green channels only', image_contour_green)
    cv2.waitKey(0)
    cv2.imwrite('green_channel.jpg', image_contour_green)
    cv2.destroyAllWindows()
    
    # detect contours using red channel and without thresholding
    contours3, hierarchy3 = cv2.findContours(image=red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_contour_red = image.copy()
    cv2.drawContours(image=image_contour_red, contours=contours3, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # see the results
    cv2.imshow('Contour detection using red channels only', image_contour_red)
    cv2.waitKey(0)
    cv2.imwrite('red_channel.jpg', image_contour_red)
    cv2.destroyAllWindows()

# link - https://learnopencv.com/contour-detection-using-opencv-python-c/
def get_gray_scale(image):
    # gray scale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # visualize the binary image
    # cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)
    # cv2.imwrite('image_thres1.jpg', thresh)
    # cv2.destroyAllWindows()
    return img_gray

def CHAIN_APPROX_SIMPLE(raw_image, gray_image):
    # 전체 keypoint 확보
    ret, thresh1 = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    image_copy2 = raw_image.copy()
    cv2.drawContours(image_copy2, contours2, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('SIMPLE Approximation contours', image_copy2)
    cv2.waitKey(0)
    cv2.imwrite('contour_point_simple.jpg', image_copy2)
    cv2.destroyAllWindows()
    image_copy3 = raw_image.copy()

    # 전체 keypoint 하나씩 확인
    # print(np.array(contours2).shape)
    # for i, contour in enumerate(contours2): # loop over one contour area
    #     print(np.array(contour).shape)
    #     for j, contour_point in enumerate(contour): # loop over the points
    #         # draw a circle on the current contour coordinate
    #         cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)
    #         # see the results
    #         cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image_copy3)
    #         cv2.waitKey(0)
    #         cv2.imwrite('contour_point_simple{}{}.jpg'.format(i,j), image_copy3)
    #         cv2.destroyAllWindows()
    #         break
    #     break


######
# Read dir
if ROOT_PATH.exists():
  for path in sorted(ROOT_PATH.rglob('*')):
    # if os.path.splitext(path)[-1] == ".jpg":
    #   img = cv2.imread(sys.path[0]+"/Images/image1.jpg",1)
    # if os.path.splitext(path)[-1] == ".bmp":
    img = cv2.imread(sys.path[0]+"/Images/pooh.bmp",1)
    break
# show_img(img)

######
# Get keypoint
# use_image_single_channel(img)
gray_image = get_gray_scale(img)
CHAIN_APPROX_SIMPLE(img, gray_image)


######
# Shape detection
# link_1 - https://stackoverflow.com/questions/64564138/how-do-i-split-up-thresholds-into-squares-in-opencv2
# link_2 - https://learnopencv.com/blob-detection-using-opencv-python-c/

# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 10;
# params.maxThreshold = 200;
 
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 1500
 
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
 
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
 
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
 
# # Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3 :
#   detector = cv2.SimpleBlobDetector(params)
# else :
#   detector = cv2.SimpleBlobDetector_create(params)


# # Detect blobs.
# keypoints = detector.detect(img)
# print(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob



