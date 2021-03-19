import utils as utils
import DBSCAN as DBSCAN
import pandas as pd
import json
import sys
import glob
import os


def format_data(cloud, labels):
    unique_labels = set(labels)

    outputData = []
    for group in unique_labels:
        groupCloud = cloud[labels == group]
        df = pd.DataFrame(groupCloud, columns=["x", "y", "z", "intensity"])
        outputData.append(df.to_dict(orient='records'))

    return outputData


if __name__ == "__main__":
    inputFolder = sys.argv[1]
    outputFolder = sys.argv[2]

    for inputPath in glob.glob(os.path.join(inputFolder, 'cloud*.json')):
        jsonCloud = utils.read_pointcloud(inputPath)
        rawCloud = utils.pointcloud_to_numpy_array(jsonCloud)

        cloud, labels = DBSCAN.run(rawCloud, eps=0.095, minSamples=9)

        outputData = format_data(cloud, labels)

        inputName = os.path.basename(inputPath)
        print("Detected {} cones in {}".format(len(set(labels)), inputName))

        outputName = inputName.replace("cloud", "cones")
        outputPath = os.path.join(outputFolder, outputName)

        with open(outputPath, 'w') as outFile:
            json.dump(outputData, outFile)

        print("Detected cones have been saved to {}".format(outputName))
