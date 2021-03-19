import utils as utils
import numpy as np
import sys
import glob
import os
import math


if __name__ == "__main__":
    sourceFolder = sys.argv[1]
    detectFolder = sys.argv[2]

    tolerance = 0.5

    totalModelCones = 0
    totalDetectedCones = 0
    truePositive = 0
    falsePositive = 0


    for sourcePath in glob.glob(os.path.join(sourceFolder, 'cones*.json')):
        sourceName = os.path.basename(sourcePath)
        detectPath = os.path.join(detectFolder, sourceName)

        jsonModelCones = utils.read_pointcloud(sourcePath)
        jsonDetectCones = utils.read_pointcloud(detectPath)

        totalModelCones += len(jsonModelCones)
        totalDetectedCones += len(jsonDetectCones)

        #print(sourceName)
        #print("\tSource cones {} ".format(len(jsonModelCones)))
        #print("\tDetected cones {}".format(len(jsonDetectCones)))

        for detectCone in jsonDetectCones:
            coneFound = False
            detectData = utils.pointcloud_to_numpy_array(detectCone)
            detectMean = np.mean(detectData, axis=0)[:3]

            for modelCone in jsonModelCones:
                modelData = utils.pointcloud_to_numpy_array(modelCone)
                modelMean = np.mean(modelData, axis=0)[:3]

                diffMean = detectMean - modelMean
                diff = math.sqrt(sum(i**2 for i in diffMean[:3]))

                if diff < tolerance:
                    coneFound = True
                    break

            if coneFound:
                truePositive += 1
            else:
                falsePositive += 1


    print("Total Model cones: {}".format(totalModelCones))
    print("Total detections: {}".format(totalDetectedCones))
    print("True positive: {}".format(truePositive))
    print("False positive: {}".format(falsePositive))