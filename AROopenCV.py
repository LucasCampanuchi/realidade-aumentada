from ast import While
from re import match
import cv2
import numpy as np

video = cv2.VideoCapture(1)
imgTarget = cv2.imread('testando.jpg')
myvid = cv2.VideoCapture('teste.mp4')

detection = False
frameCounter = 0

success, imgVideo = myvid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures = 1000)
orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
orb.setWTA_K(3)

kp1, des1 = orb.detectAndCompute(imgTarget, None)
imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)


while True:
  sucess, frame = video.read()
  imgAug = frame.copy()

  kp2, des2 = orb.detectAndCompute(frame, None)
  frame = cv2.drawKeypoints(frame, kp2, None)

  if detection == False:
    myvid.set(cv2.CAP_PROP_POS_FRAMES, 0)
  else:
    if frameCounter == myvid.get(cv2.CAP_PROP_FRAME_COUNT):
      myvid.set(cv2.CAP_PROP_POS_FRAMES, 0)
      frameCounter = 0

    success, imgVideo = myvid.read()
    imgVideo = cv2.resize(imgVideo, (wT, hT))
    
  

  if des2 is not None:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    if len(matches[0]) > 1:
      for m,n in matches:
        if m.distance < 0.75 * n.distance:
          good.append(m)

      imgFeatures = cv2.drawMatches(imgTarget, kp1, frame, kp2, good, None, flags = 2)

      if(len(good) > 5):
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        if matrix is not None:
          pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
          dst = cv2.perspectiveTransform(pts, matrix)
          img2 = cv2.polylines(frame, [np.int32(dst)], True, (255,0,255), 3)

          imgWarp = cv2.warpPerspective(imgVideo, matrix, (frame.shape[1], frame.shape[0]))

          maskNew = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
          cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
          maskInv = cv2.bitwise_not(maskNew)
          imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv)
          imgAug = cv2.bitwise_or(imgWarp, imgAug)

          cv2.imshow('maskNew', imgAug)

  cv2.imshow('Video', frame)
  cv2.waitKey(1)
  frameCounter += 1