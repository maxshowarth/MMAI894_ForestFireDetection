import os, shutil
import random

# Get all names of normal tiles into a list
unsortedTileDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/tiles/") # Tiles that are unsorted are normal
selectedNormalTileDir = os.path.dirname("/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/sorted_images/selected_normal/")
normalTileFileNames = []

for tile in os.listdir(unsortedTileDir):
    if tile.endswith(".png"):
        normalTileFileNames.append(tile)

# Select 5000 random normal images to be used in training/validation

selectedNormalTileNames = random.sample(normalTileFileNames, 5000)

for filename in selectedNormalTileNames:
    newPath = os.path.join(selectedNormalTileDir,filename)
    shutil.move(os.path.join(unsortedTileDir, filename), newPath)


