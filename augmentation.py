import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator


# Load images
def load_data(folder_path=None, folder=None):
    """
        Load the data for image augmentation:
        :param folder_path: path to original folder
        :type folder: folder name

    """

    dataset = folder_path
    images = []

    # Iterate through each image in our folder
    for file in os.listdir(os.path.join(dataset, folder)):
        # Get the path name of the image
        img_path = os.path.join(os.path.join(dataset, folder), file)

        curr_img = cv2.imread(img_path)

        images.append(curr_img)

    images = np.array(images, dtype='float32')

    return images


# Augmentation
def augmentation(images, save_to_dir=None, save_prefix=None, total=100):
    if os.path.isdir(save_to_dir):
        pass
    else:
        os.makedirs(save_to_dir)
    n = 0
    data_gen_args = dict(featurewise_center=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         shear_range=0.2
                         )

    datagen = ImageDataGenerator(**data_gen_args)
    datagen.fit(images)
    imagegen = datagen.flow(images, batch_size=1, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format='png')

    for image in imagegen:
        n += 1
        if total == n:
            break


# load original fire images and save augmented images in local folder
org_images = load_data(folder_path='images/train', folder='fire')
augmentation(org_images, save_to_dir='augmented_images_fire', save_prefix='aug', total=100)
