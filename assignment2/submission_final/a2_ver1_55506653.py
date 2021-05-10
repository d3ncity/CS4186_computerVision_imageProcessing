# Name: Denny-Thomas-Varghese
# SID: 55506653

import cv2 as cv
import numpy as np

# OpenCV to read an image in grayscale
def readImgGrayScale(imgURL):
    return cv.imread(imgURL, cv.IMREAD_GRAYSCALE)

def constructDisparityMap(whichPair):
    # To make up the paths, use this format: StereoMatchingTestings/whichPair/viewNo
    # StereoMatchingTestings contains all the 3 folders obtained from the mentioned Google Drive Link in the prompt doc
    # Initialize variables to make up the paths to imgs
    folderPath = "./StereoMatchingTestings/"
    view1 = "/view1.png"
    view2 = "/view5.png"

    resultFolderPath = "allResults/"

    # openCV to read img in grayscale
    print("Reading the two images...")
    try:
        print("Reading from: " + folderPath+whichPair+view1, cv.IMREAD_GRAYSCALE)
        img1 = readImgGrayScale(folderPath+whichPair+view1)
        print("and Reading from: " + folderPath+whichPair+view2, cv.IMREAD_GRAYSCALE)
        img2 = readImgGrayScale(folderPath+whichPair+view2)
    except:
        print("Both or either of the images could not be read. Please double-check the paths to the image files.")
        exit()
    print("Both images have been successfully read.")


    # Folder to store current iteration's result
    allCurrentResultDirectory = resultFolderPath + "others/" + whichPair # to store any results other than disparity maps
    dispCurrentResultDirectory = resultFolderPath + "pred/" + whichPair # to store diparity maps

    # Initializing SIFT detector
    sift = cv.SIFT_create()
    # extract the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Visualize keypoints
    imgSift = cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("allKeypointsFromSIFT", imgSift)
    cv.imwrite(allCurrentResultDirectory + "/allKeypointsFromSIFT.png", imgSift)

    # Match keypoints in both images
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    allMatches = flann.knnMatch(des1, des2, k=2)

    mask = [[0, 0] for i in range(len(allMatches))]
    goodMatches = []
    pts1 = []
    pts2 = []

    #filter out the bad matches and only keep the good matches (less than threshold)
    for i, (m, n) in enumerate(allMatches):
        if m.distance < 0.7*n.distance:
            mask[i] = [1, 0]
            goodMatches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Plot the matching keypoints from one image to the other
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask[300:500],flags=cv.DrawMatchesFlags_DEFAULT)
    keypoint_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, allMatches[300:500], None, **draw_params)
    cv.imshow("keypointMatchesFromFLANN", keypoint_matches)
    cv.imwrite(allCurrentResultDirectory + "/keypointMatchesFromFLANN.png", keypoint_matches)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # find fundamental matrix
    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    # select only inliers after RANSAC
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)

    # get image-sizes from shape
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    # Stereo rectification - the Uncalibrated Variant
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    # Rectify (undistort) the images and save them
    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
    cv.imwrite(allCurrentResultDirectory+"/rectifiedImg1.png", img1_rectified)
    cv.imwrite(allCurrentResultDirectory+"/rectifiedImg2.png", img2_rectified)

    bSize = 11
    minDisp = -128
    maxDisp = 128
    nDisp = maxDisp - minDisp
    disp12MaxDiff = 0
    uRatio = 5
    speckleWindowSize = 200
    speckleRange = 2

    stereo1 = cv.StereoSGBM_create(
        minDisparity=minDisp,
        numDisparities=nDisp,
        blockSize=bSize,
        uniquenessRatio=uRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * bSize * bSize,
        P2=32 * 1 * bSize * bSize,
    )

    disparity_SGBM = stereo1.compute(img1_rectified, img2_rectified)

    disparity_SGBM = stereo1.compute(img1_rectified, img2_rectified)
    
    matcher_img2 = cv.ximgproc.createRightMatcher(stereo1)
    disp2 = matcher_img2.compute(img2_rectified, img1_rectified)
    
    lmb = 35000
    sigma = 1.5

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo1)
    wls_filter.setLambda(lmb)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(disparity_SGBM, img1_rectified, disparity_map_right=disp2)
    filtered_disp = cv.normalize(filtered_disp, filtered_disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    cv.imwrite(dispCurrentResultDirectory + "/disp1.png", filtered_disp)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__=="__main__":

    pairNames = ["Art", "Dolls", "Reindeer"]
    for pairName in pairNames:
        print("Computing for {}".format(pairName))
        constructDisparityMap(pairName)

