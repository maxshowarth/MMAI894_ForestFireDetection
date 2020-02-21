import pickle as pk
import os, shutil
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
print(os.getcwd())
os.chdir(os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/"))
print(os.getcwd())
smokeDir = os.path.dirname("/sorted_images/smoke/")
unsureDir = os.path.dirname("/sorted_images/unsure/")
clearDir = os.path.dirname("/sorted_images/clear/")
sourceDir = os.path.dirname("/code_cache/processedImages/")

tileInfo = pk.load(open("/code_cache/tile_infoPickle", 'rb'))
columns = ["sourceFile", "tileName", "tilePath"]
tileInfo_df = pd.DataFrame(columns = columns)
moved_tiles = []
tileSize = 234

for sourceImage in tileInfo:
    sourceTileInfo = pd.DataFrame.from_records(sourceImage)
    sourceTileInfo.columns = columns
    tileInfo_df = tileInfo_df.append(sourceTileInfo, ignore_index=True)
global movePath
movePath = ""

for sourceImage in tileInfo_df['sourceFile'].unique():
    cv2.destroyAllWindows()
    print(sourceImage)
    sourceImageTiles = tileInfo_df[tileInfo_df['sourceFile'] == sourceImage]
    cv2.namedWindow("Source_Image")
    cv2.namedWindow("Tile")
    for index, tile in sourceImageTiles.iterrows():
        cv2.destroyAllWindows()
        movePath = ""
        tilePath = os.path.normpath(tile['tilePath'])
        tile_img = cv2.imread(tilePath)
        cv2.imshow("Tile", tile_img)
        cv2.waitKey(1)
        while movePath == "":
            print(tile['tileName'])
            kb_input = input("Enter choice: ")
            if kb_input == 'f':
                movePath = "moved to fire"
                newPath = os.path.join(smokeDir, tile['tileName'])
                shutil.move(tile['tilePath'], newPath)
                print(movePath)
                continue
            if kb_input == 'p':
                sourceImagePath = os.path.join(sourceDir, "{}_prc".format(tile['sourceFile']))
                sourceImageLoaded = cv2.imread(sourceImagePath)
                splitName = tile['tileName'].split("-")
                rectImage = cv2.rectangle(sourceImageLoaded, (int(splitName[1]), int(splitName[2])), (tileSize, tileSize), (255, 0, 0), 15)
                height, width, channels = rectImage.shape
                rectImageSmall = cv2.resize(rectImage, (height // 10, width // 10))
                cv2.imshow('Source_Image', rectImageSmall)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print("no move")
            if kb_input == 's':
                print("skip")
                movePath = "skipped"
                print(movePath)
                print("test")
