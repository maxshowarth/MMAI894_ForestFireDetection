import pickle as pk
import os, shutil
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import xml.etree.ElementTree as ET
import xmltodict

os.chdir(os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/"))
codeCacheDir = os.path.dirname("/code_cache/")
cloudDir = os.path.dirname("sorted_images/cloud/")
fireDir = os.path.dirname("sorted_images/fire/")
smokeDir = os.path.dirname("sorted_images/smoke/")
unsureDir = os.path.dirname("sorted_images/unsure/")
clearDir = os.path.dirname("sorted_images/clear/")
sourceDir = os.path.dirname("code_cache/processedImages/")
annotationsDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/rectLabel Annotations/")
unsortedTileDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/tiles/")

tileInfo = pk.load(open("code_cache/tile_infoPickle", 'rb'))
columns = ["sourceFile", "tileName", "originalTilePath","xmin", "ymin", "xmax", "ymax"]
tileInfo_df = pd.DataFrame(columns = columns)
moved_tiles = []
tileSize = 234

# Load tile info into dataframe
for sourceImage in tileInfo:
    for tile in sourceImage:
        splitName = tile[1].split("-")
        tile.extend([splitName[1],splitName[2], splitName[3], splitName[4].split(".")[0]] )
        sourceTileInfo = pd.DataFrame.from_records([tile])
        sourceTileInfo.columns = columns
        tileInfo_df = tileInfo_df.append(sourceTileInfo, ignore_index=True)


# Get bounding boxes from XML files
boundingBoxes = pd.DataFrame(columns=["sourceFile", "label", "xmin", "ymin", "xmax", "ymax" ])
for file in os.listdir(annotationsDir):
    if file.endswith(".xml"):
        filePath = os.path.join(annotationsDir, file)
        with open(filePath) as f_xml:
            f = xmltodict.parse(f_xml.read())
            filename = f["annotation"]['filename']
            for row in f['annotation']['object']:
                rowIndex = len(boundingBoxes)+1
                boundingBoxes.loc[rowIndex]  = [filename, row['name'], row['bndbox']["xmin"], row['bndbox']["ymin"], row['bndbox']["xmax"], row['bndbox']['ymax']]
                # print([filename, row['name'], row['bndbox']["xmin"], row['bndbox']["ymin"], row['bndbox']["xmax"], row['bndbox']['ymax']])

# Add newTilePath columns to DataFrame
# TODO: need to modify this so that it can be run when new tiles are added without deleting the paths of the already sorted tiles
tileInfo_df["label"] = ""
tileInfo_df["newPath"] = ""


# Get catalogue available tile images:
unsortedTiles = []
for file in os.listdir(unsortedTileDir):
    if file.endswith(".png"):
        unsortedTiles.append(file)

# Convert numerical columns to int
boundingBoxes['xmin'] = boundingBoxes['xmin'].astype(int)
boundingBoxes['ymin'] = boundingBoxes['ymin'].astype(int)
boundingBoxes['xmax'] = boundingBoxes['xmax'].astype(int)
boundingBoxes['ymax'] = boundingBoxes['ymax'].astype(int)
tileInfo_df['xmin'] = tileInfo_df['xmin'].astype(int)
tileInfo_df['ymin'] = tileInfo_df['ymin'].astype(int)
tileInfo_df['xmax'] = tileInfo_df['xmax'].astype(int)
tileInfo_df['ymax'] = tileInfo_df['ymax'].astype(int)


# Start comparing uncatalogued tiles to bounding boxes
for tile in unsortedTiles:
    thisTileInfo = tileInfo_df[tileInfo_df["tileName"] == tile].to_dict('r')[0]
    boundingBoxesHolder = boundingBoxes.loc[boundingBoxes["sourceFile"]=="{}_prc.tif".format(thisTileInfo["sourceFile"])]

    # Old overlapping method that looks for explicit overlaps but only covers quadrant 1 overlaps
    # overlappingBoundingBoxes = boundingBoxesHolder[ (boundingBoxesHolder["xmin"]<thisTileInfo["xmin"]) &
    #                                                 (boundingBoxesHolder["xmax"]>thisTileInfo["xmax"]) &
    #                                                 (boundingBoxesHolder["ymin"]<thisTileInfo["ymin"]) &
    #                                                 (boundingBoxesHolder["ymax"]>thisTileInfo["ymax"])]

    # First step is determine which bounding boxes DO NOT overlap tiles
    overlappingBoundingBoxes = boundingBoxesHolder[(boundingBoxesHolder["xmin"]>thisTileInfo["xmax"])| # Check if bounding box is below tile
                                                   (boundingBoxesHolder["xmax"]<thisTileInfo["xmin"])| # Check if bounding box is above tile
                                                   (boundingBoxesHolder["ymin"]>thisTileInfo["ymax"])| # Check if bounding box is right of tile
                                                   (boundingBoxesHolder["ymax"]<thisTileInfo["ymin"])] # Check if bounding box is left of tile
    print("{} bounding boxes DO NOT overlap out of {} boxes in this source image".format(len(overlappingBoundingBoxes), len(boundingBoxesHolder)))
    # Remove bounding boxes that do not overlap tiles
    overlappingBoundingBoxes = pd.concat([boundingBoxesHolder,overlappingBoundingBoxes]).drop_duplicates(keep=False)
    print("{} bounding boxes DO overlap out of {} boxes in this source image".format(len(overlappingBoundingBoxes), len(boundingBoxesHolder)))
    # If there is at least 1 overlapping bounding box, assign the label to the tile and move to the appropraite directory
    if len(overlappingBoundingBoxes)>0:
        label = overlappingBoundingBoxes.to_dict('r')[0]['label']
        tileInfo_df.loc[tileInfo_df["tileName"] == tile,"label"] = label
        if label == "fire":
            movePath = "moved to fire"
            newPath = os.path.join(fireDir, tile)
            shutil.move(thisTileInfo['originalTilePath'], newPath)
            tileInfo_df.loc[tileInfo_df["tileName"] == tile, "newPath"] = newPath
            print(movePath)
        if label == "smoke":
            movePath = "moved to smoke"
            newPath = os.path.join(smokeDir, tile)
            shutil.move(thisTileInfo['originalTilePath'], newPath)
            tileInfo_df.loc[tileInfo_df["tileName"] == tile, "newPath"] = newPath
            print(movePath)
        if label == "cloud":
            movePath = "moved to cloud"
            newPath = os.path.join(cloudDir, tile)
            shutil.move(thisTileInfo['originalTilePath'], newPath)
            tileInfo_df.loc[tileInfo_df["tileName"] == tile, "newPath"] = newPath
            print(movePath)

pk.dump(tileInfo_df, open(os.path.join(codeCacheDir,"tile_info_sorted_Pickle"), "wb") )



print("done")