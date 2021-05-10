import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt


def constructDisparityMap(whichPairValue):

    # To make up the paths, use this format: StereoMatchingTestings/whichPair/viewNo

    # allDataFromDriveLink contains all the 3 folders obtained from the mentioned Google Drive Link in the prompt doc


    # Initialize variables to make up the paths to imgs


    folderPath = "./StereoMatchingTestings/"

    whichPair = whichPairValue #change to "Dolls" or "Reindeer"

    view1 = "/view1.png"

    view2 = "/view5.png"


    # openCV to read img in grayscale

    print("Reading the two images...")

    try:

        print("Reading from: " + folderPath+whichPair+view1, cv.IMREAD_GRAYSCALE)
        img1 = cv.imread(folderPath+whichPair+view1, cv.IMREAD_GRAYSCALE)
        print("and Reading from: " + folderPath+whichPair+view2, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(folderPath+whichPair+view2, cv.IMREAD_GRAYSCALE)

    except:

        print("Both or either of the images could not be read. Please double-check the paths to the image files.")
        exit()

    print("Both images have been successfully read.")


    # Folder to store current iteration's result
    allCurrentResultDirectory = "allResults/everythingElse/" + whichPair
    dispCurrentResultDirectory = "allResults/pred/" + whichPair

    print("Results stored in " + allCurrentResultDirectory)

    # PREPROCESSING
    # Compare unprocessed images

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    axes[0].imshow(img1, cmap="gray")

    axes[1].imshow(img2, cmap="gray")

    axes[0].axhline(250)

    axes[1].axhline(250)

    axes[0].axhline(450)

    axes[1].axhline(450)

    plt.suptitle("Original images")

    plt.savefig(allCurrentResultDirectory + "/original_images.png")

    plt.show()


    sift = cv.SIFT_create() # Initializing SIFT detector

    # find the keypoints and descriptors with SIFT

    kp1, des1 = sift.detectAndCompute(img1, None)

    kp2, des2 = sift.detectAndCompute(img2, None)


    # Visualize keypoints

    imgSift = cv.drawKeypoints(

        img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("SIFT Keypoints", imgSift)

    cv.imwrite(allCurrentResultDirectory + "/sift_keypoints.png", imgSift)


    # Match keypoints in both images

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)


    # Keep good matches: calculate distinctive image features

    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []

    pts1 = []

    pts2 = []


    for i, (m, n) in enumerate(matches):

        if m.distance < 0.7*n.distance:

            # Keep this keypoint pair

            matchesMask[i] = [1, 0]
            good.append(m)

            pts2.append(kp2[m.trainIdx].pt)

            pts1.append(kp1[m.queryIdx].pt)


    # Draw the keypoint matches between both pictures

    # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    draw_params = dict(matchColor=(0, 255, 0),

                    singlePointColor=(255, 0, 0),

                    matchesMask=matchesMask[300:500],

                    flags=cv.DrawMatchesFlags_DEFAULT)


    keypoint_matches = cv.drawMatchesKnn(

        img1, kp1, img2, kp2, matches[300:500], None, **draw_params)

    cv.imshow("Keypoint matches", keypoint_matches)

    cv.imwrite(allCurrentResultDirectory + "/keypoint_matches.png", keypoint_matches)



    # STEREO RECTIFICATION

    pts1 = np.int32(pts1)

    pts2 = np.int32(pts2)

    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)


    # We select only inlier points

    pts1 = pts1[inliers.ravel() == 1]

    pts2 = pts2[inliers.ravel() == 1]


    # Visualize epilines

    # Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html



    def drawlines(img1src, img2src, lines, pts1src, pts2src):

        ''' img1 - image on which we draw the epilines for the points in img2

            lines - corresponding epilines '''

        r, c = img1src.shape

        img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)

        img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)

        # Edit: use the same random seed so that two images are comparable!

        np.random.seed(0)

        for r, pt1, pt2 in zip(lines, pts1src, pts2src):

            color = tuple(np.random.randint(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -r[2]/r[1]])

            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])

            img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)

            img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)

            img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)

        return img1color, img2color

    lines1 = cv.computeCorrespondEpilines(

        pts2.reshape(-1, 1, 2), 2, fundamental_matrix)

    lines1 = lines1.reshape(-1, 3)

    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(

        pts1.reshape(-1, 1, 2), 1, fundamental_matrix)

    lines2 = lines2.reshape(-1, 3)

    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)


    plt.subplot(121), plt.imshow(img5)

    plt.subplot(122), plt.imshow(img3)

    plt.suptitle("Epilines in both images")

    plt.savefig(allCurrentResultDirectory + "/epilines.png")

    plt.show()

    # Stereo rectification - Uncalibrated Variant

    h1, w1 = img1.shape

    h2, w2 = img2.shape

    _, H1, H2 = cv.stereoRectifyUncalibrated(

        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    # Rectify (undistort) the images and save them

    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))

    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))

    cv.imwrite(allCurrentResultDirectory+"/rectified_1.png", img1_rectified)

    cv.imwrite(allCurrentResultDirectory+"/rectified_2.png", img2_rectified)


    # Draw the rectified images

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    axes[0].imshow(img1_rectified, cmap="gray")

    axes[1].imshow(img2_rectified, cmap="gray")

    axes[0].axhline(250)

    axes[1].axhline(250)

    axes[0].axhline(450)

    axes[1].axhline(450)

    plt.suptitle("Rectified images")

    plt.savefig(allCurrentResultDirectory + "/rectified_images.png")

    plt.show()


    # ------------------------------------------------------------

    # CALCULATE DISPARITY (DEPTH MAP)
    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.

    block_size = 11

    min_disp = -128

    max_disp = 128

    # Maximum disparity minus minimum disparity. The value is always greater than zero.

    # In the current implementation, this parameter must be divisible by 16.

    num_disp = max_disp - min_disp

    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.

    # Normally, a value within the 5-15 range is good enough

    uniquenessRatio = 5

    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.

    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.

    speckleWindowSize = 200

    # Maximum disparity variation within each connected component.

    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.

    # Normally, 1 or 2 is good enough.

    speckleRange = 2

    disp12MaxDiff = 0


    stereo = cv.StereoSGBM_create(

        minDisparity=min_disp,

        numDisparities=num_disp,

        blockSize=block_size,

        uniquenessRatio=uniquenessRatio,

        speckleWindowSize=speckleWindowSize,

        speckleRange=speckleRange,

        disp12MaxDiff=disp12MaxDiff,

        P1=8 * 1 * block_size * block_size,

        P2=32 * 1 * block_size * block_size,
    )

    disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)


    plt.imshow(disparity_SGBM, cmap='plasma')

    plt.colorbar()

    plt.show()


    # Normalize the values to a range from 0..255 for a grayscale image

    disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,

                                beta=0, norm_type=cv.NORM_MINMAX)

    disparity_SGBM = np.uint8(disparity_SGBM)

    cv.imshow("Disparity", disparity_SGBM)

    cv.imwrite(dispCurrentResultDirectory + "/disp1.png", disparity_SGBM)



    cv.waitKey()

    cv.destroyAllWindows()

    # ---------------------------------------------------------------


if __name__=="__main__":

    whichPairValues = ["Art", "Dolls", "Reindeer"]

    for whichPairValue in whichPairValues:

        print("Computing for the {} pair".format(whichPairValue))

        constructDisparityMap(whichPairValue)

