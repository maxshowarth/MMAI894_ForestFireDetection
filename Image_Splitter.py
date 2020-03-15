from PIL import Image
import numpy as np
import argparse
import cv2
from skimage import io
from skimage.external import tifffile
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from collections import defaultdict
import os
from time import strftime
from datetime import datetime

masterDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/")
rawImageDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/visual_images_and_metadata/")
codeCacheDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/")
tileCacheDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/tiles/")
baseImagePath = "/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/visual_images_and_metadata/LC80890822019310LGN00_Visual.tif"

def imageSplit(imageDir, imageName, tileSize):
    startTime = datetime.now()
    print("\t\tBeginning split for image {} at {}".format(imageName, startTime.strftime("%H:%M:%S")))
    imagePath = os.path.join(imageDir, imageName)
    # Create DataFrame to store image information
    tileInfo = []

    # Convert image to PNG for processing
    baseImage = Image.open(imagePath)
    baseImage.save(os.path.join(codeCacheDir, "baseImage_PNG.png"), "PNG") #/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/baseImgae_PNG", "PNG")
    baseImage = io.imread(os.path.join(codeCacheDir, "baseImage_PNG.png"))
    conversionTime = datetime.now() - startTime
    print("\t\t\t Conversion completed in {} seconds".format(conversionTime.seconds))

    # Rotate and save image
    grayscale = rgb2gray(baseImage)
    angle = determine_skew(grayscale)
    rotated = rotate(baseImage, angle, resize=True)*255
    io.imsave(os.path.join(codeCacheDir,"baseImgae_PNG_rotated.png"), rotated.astype(np.uint8))
    rotateTime = datetime.now() - startTime + conversionTime
    print("\t\t\t Rotation completed in {} seconds".format(rotateTime.seconds))

    # Find and crop grey edges
    img = cv2.imread(os.path.join(codeCacheDir,"baseImgae_PNG_rotated.png"))
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = contours[0]
    x, y, w, h = cv2.boundingRect(count)
    cropped = img[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(codeCacheDir, "grayscale_check.png"), grayscale)
    cv2.imwrite(os.path.join(codeCacheDir,"baseImage_rotate_borderless.png"), cropped)
    borderRemovalTime = datetime.now() - startTime + rotateTime
    print("\t\t\t Border removal completed in {} seconds".format(borderRemovalTime.seconds))

    # Determine chunk size and chunk images
    ready_image = Image.open(os.path.join(codeCacheDir,"baseImage_rotate_borderless.png"))
    width, height = ready_image.size
    chunks_wide = width // tileSize
    chunks_high = height // tileSize
    # ready_image = io.imread('/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/baseImage_rotate_borderless.png')

    chunkCoords = []
    chunkCount = (chunks_high + 1) * (chunks_wide + 1)
    chunkDim = 234
    Xs = []
    Ys = []
    increasingX = 0
    increasingY = 0
    while len(Xs) < chunks_wide:
        Xs.append(increasingX)
        increasingX += chunkDim
    Xs.append(width - chunkDim)

    while len(Ys) < chunks_high:
        Ys.append(increasingY)
        increasingY += chunkDim
    Ys.append(height - chunkDim)

    for x in Xs:
        x1 = x
        x2 = x + chunkDim
        for y in Ys:
            y1 = y
            y2 = y + chunkDim
            chunkCoords.append([x1, y1, x2, y2])

    # Generate Tiles

    for tile in chunkCoords:
        cropped = ready_image.crop((tile[0], tile[1], tile[2], tile[3]))
        filename = "{}-{}-{}-{}-{}.png".format(imageName, tile[0], tile[1], tile[2], tile[3])
        filepath = os.path.join(tileCacheDir, filename)
        tileInfo.append([imageName,filename, filepath])
        cropped.save(filepath, "PNG")
    tileGenerationTime = datetime.now() - startTime + borderRemovalTime
    print("\t\t\t Tile generation completed in {} seconds".format(tileGenerationTime.seconds))
    ready_image.save(os.path.join(codeCacheDir, "processedImages/{}_prc".format(imageName)), "PNG")
    totalTime = datetime.now() - startTime

    print("\t\tFile: {} completed in {} seconds\r\r".format(imageName, totalTime.seconds))

    return tileInfo

# tileInfo = imageSplit(rawImageDir,"LC80890822019310LGN00_Visual.tif")

