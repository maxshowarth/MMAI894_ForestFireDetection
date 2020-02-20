import os, sys, traceback, threading
from time import strftime
from datetime import datetime
from skimage import io
from Image_Splitter import imageSplit
import pickle

masterDir = os.path.dirname("//")
rawImageDir = os.path.dirname("/visual_images_and_metadata/")
codeCacheDir = os.path.dirname("/code_cache/")
tileCacheDir = os.path.dirname("/code_cache/tiles/")
baseImagePath = "/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/visual_images_and_metadata/LC80890822019310LGN00_Visual.tif"

tileSize = 234
tile_info=[]

# baseImage = io.imread(r"/Users/max/Quick Jupyter Notebooks/MMAI/MMAI 894 - Deep Learning/code_cache/baseImage_PNG")
for file in os.listdir(rawImageDir):
    print("Examining file: {}".format(file))
    if file.endswith(".tif"):
        print("\tSplitting image")
        tile_info.append(imageSplit(rawImageDir, file, tileSize))
    else:
        continue
pickle.dump(tile_info, open(os.path.join(codeCacheDir,"tile_infoPickle"), "wb"))
print("all files examined")
sys.exit()