from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

MAX_IMAGE_COUNT = 5
MAX_QUERYIMAGE_COUNT = 2

basePathImg = "./Images/"
basePathQueryImg = "./Queries/"
allDesc = {}
allFeatures = {}
allHistograms = {}
sift_distances = {}
# helper function used by all methods
# imgURL is of the format "./folder/name.jpg", .txt for the coordinates (if it exists), isCNN=True if resizing to (244, 244) is required
# if .txt does not exist, return entire img
def extractInstance(imgURL, isCNN):
    img = None
    try:
        print("isCNN=" + str(isCNN))
        print("Getting image or instance: " + imgURL)
        img = cv2.imread(imgURL)

        imgCoordinatesPath = imgURL[:-4] + ".txt"
        with open(imgCoordinatesPath) as coordinatesFile:
            line = coordinatesFile.readline()[:-1]
            coordinates = line.split(" ")
            instanceImg = img[int(coordinates[1]):int(coordinates[1])+int(coordinates[3]), int(coordinates[0]):int(coordinates[0])+int(coordinates[2])]
            if(isCNN):
                instanceImg = cv2.resize(instanceImg, (244, 244))
            return instanceImg
    except IOError:
        print("instance dimensions absent, returning img itself")
        if(isCNN):
            img = cv2.resize(img, (244, 244))
        return img

def extractGrayInstance(imgURL):
    img = None
    try:
        img = cv2.imread(imgURL, 0)

        imgCoordinatesPath = imgURL[:-4] + ".txt"
        with open(imgCoordinatesPath) as coordinatesFile:
            line = coordinatesFile.readline()[:-1]
            coordinates = line.split(" ")
            instanceImg = img[int(coordinates[1]):int(coordinates[1])+int(coordinates[3]), int(coordinates[0]):int(coordinates[0])+int(coordinates[2])]
            return instanceImg
    except IOError:
        return img

def numpyArraysEuclidianDist(a1, a2):
    dist = np.sum([(x2-x1)**2 for (x1, x2) in zip(a1, a2)]) **0.5
    return dist

#helper function for color_hist(), the color histogram function - returns numpy array of color histogram using 16 bins
def create_hist_bgr(image):
    hist_bgr = np.zeros([16 * 16 * 16, 1], np.float32)
    height, width, channels = image.shape #for iterating through
    bin_size = 256 / 16 #using 16 bins
    for rowCount in range(height):
        for colCount in range(width):
            b = image[rowCount, colCount, 0]
            g = image[rowCount, colCount, 1]
            r = image[rowCount, colCount, 2]
            index = int(b / bin_size) * 16 * 16 + int(g / bin_size) * 16 + int(r / bin_size)
            hist_bgr[int(index), 0] += 1

    return hist_bgr

def color_hist():
    #Database Images
    for i in range(1, MAX_IMAGE_COUNT + 1):
        img = extractInstance(basePathImg + str(i).zfill(4) +".jpg", False)
        img_hist = create_hist_bgr(img)
        allHistograms[i] = img_hist
    print("****Completion stmt: All Database Images' histograms extracted and stored")

    f = open("rankList.txt", "a")
    #Query Images
    for k in range (1, MAX_QUERYIMAGE_COUNT + 1):
        histogram_distances = {} #clear histogram distances for every query
        histogram_allItems = "" #clear the list to be appended to rankList for every query
        qImg = extractInstance(basePathQueryImg + str(k).zfill(2) + ".jpg", False)
        qImg_histogram = create_hist_bgr(qImg)

        for j in range(1, MAX_IMAGE_COUNT + 1):
            hist_dist = numpyArraysEuclidianDist(qImg_histogram, allHistograms[i])
            histogram_distances[j] = hist_dist

        histogram_distances = {k: v for k, v in sorted(histogram_distances.items(), key=lambda item: item[1])} #sort in ascending order of distances (items)

        for key, d in histogram_distances.items():
            histogram_allItems += str(key) +" " # list of simliar images in order - to append to the rankList.txt file
        toAppend = ("Q{}: {}".format(k, histogram_allItems))
        f.write(toAppend)
        f.write("\n")
    f.close()

def printAllDesc():
    for x in allDesc.keys():
        print(x)
        print("\n")
        print(allDesc[x])

def cnn():
    model = ResNet50(weights='imagenet', include_top=False)
    model.summary()
    #Database Images
    for i in range(1, MAX_IMAGE_COUNT + 1):
        img = extractInstance(basePathImg + str(i).zfill(4) +".jpg", True)

        #preprocessing for predict() input
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        resnet50_feature = model.predict(img_data)
        resnet50_feature = resnet50_feature.squeeze()
        allFeatures[i] = resnet50_feature
    print("****Completion stmt: All Database Images' features extracted and stored")

    f = open("rankList.txt", "a")
    #Query Images
    for k in range (1, MAX_QUERYIMAGE_COUNT + 1):
        cnn_distances = {}
        cnn_allItems = ""
        qImg = extractInstance(basePathQueryImg + str(k).zfill(2) + ".jpg", True)
        qImg_data = image.img_to_array(qImg)
        qImg_data = np.expand_dims(qImg_data, axis=0)
        qImg_data = preprocess_input(qImg_data)
        q_resnet50_feature = model.predict(qImg_data)
        q_resnet50_feature = q_resnet50_feature.squeeze()

        for j in range(1, MAX_IMAGE_COUNT + 1):
            dist = np.linalg.norm(q_resnet50_feature-allFeatures[j])
            cnn_distances[j] = dist

        cnn_distances = {k: v for k, v in sorted(cnn_distances.items(), key=lambda item: item[1])}

        for key, d in cnn_distances.items():
            cnn_allItems += str(key) +" "
        toAppend = ("Q{}: {}".format(k, cnn_allItems))
        f.write(toAppend)
        f.write("\n")
    f.close()

# The Combination Method SIFT and BRISK
def Sift_and_Brisk():
    f = open("rankList.txt", "a")
    for i in range(1, MAX_QUERYIMAGE_COUNT + 1):
        sift_distances = {}
        sift_allItems = ""
        qImg = extractGrayInstance(basePathQueryImg + str(i).zfill(2) +".jpg")
        for j in range(1, MAX_IMAGE_COUNT + 1):
            img = extractGrayInstance(basePathImg + str(i).zfill(4) +".jpg")
            calcSiftBriskDist(qImg, img, i, j)

        sift_distances = {k: v for k, v in sorted(sift_distances.items(), key=lambda item: item[1])}

        for key, d in sift_distances.items():
            sift_allItems += str(key) +" "
        toAppend = ("Q{}: {}".format(i, sift_allItems))
        f.write(toAppend)
        f.write("\n")
    f.close()

def calcSiftBriskDist(img, qImg, qImgKey, imgKey):

    # create instances
    siftDetector = cv2.xfeatures2d.SIFT_create()
    brisk = cv2.BRISK_create()

    #extract descriptors SIFT
    kpImg, desImg = siftDetector.detectAndCompute(img, None)
    kpQuery, desQuery = siftDetector.detectAndCompute(qImg, None)

    # extract descriptors BRISK
    kpImgBrisk, descImgBrisk = brisk.detectAndCompute(img, None)
    kpQImgBrisk, descQImgBrisk = brisk.detectAndCompute(qImg, None)

    if(desImg is None or desQuery is None or descImgBrisk is None or descQImgBrisk is None):
        dist = 99999999
        sift_distances[imgKey] = dist
        return dist

    # flatten to solve shape issues and append
    descQImg = np.append(desQuery.flatten(), descQImgBrisk.flatten())
    descImg = np.append(desImg.flatten(), descImgBrisk.flatten())

    maxlen = max(len(descQImg),len(descImg))

    # append
    descQImg = np.concatenate((descQImg , np.zeros(maxlen-len(descQImg))))
    descImg = np.concatenate((descImg , np.zeros(maxlen-len(descImg))))

    #euclidian distance
    dist = np.linalg.norm(descQImg - descImg)

    # store in dictionary
    sift_distances[imgKey] = dist

if __name__ == "__main__":
    cnn() # the best method in terms of precision
