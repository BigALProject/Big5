"""
  filter_images.py
  SEIS764 - Artificial Intelligence

  Takes an input directory of images and filters based on filename and resulting mask size.
"""
import os
import re
import cv2
import shutil
import pickle
import pandas as pd

# Path to Zach's post processed masked images.
mask_image_location = "../../data/maskedImage/"
# Cache file
cache_file = "data.cache"
# Filter for destruction type.  Set to "" for all destruction types.
filter_destruction_type = "tornado"
# Minimum footprint - too small of images will likely cause issues, so lets skip them.
# This is just used to find really small footprints.  Use filter_min_footprint instead.
notify_min_footprint = 50
# Minimum footprint for dataset.
filter_min_footprint = 3000
# Maximum footprint for dataset.
filter_max_footprint = 9999999
# Location of the output directory for filtered images
output_dir = "../../data/filtered_images_for_training/"


def extract_non_empty_pixel_count(filename):
    '''
    Find out how many pixels are in the masked image that are NOT empty.
    :param filename: Name of file to process.
    :return: 2D array of [count of non black pixels, and the filename].
    '''
    image = cv2.imread(filename, 0)
    count = cv2.countNonZero(image)
    return count


if __name__ == "__main__":

    total = 0
    cache_info = []

    print("Creating output directories in output_dir=" + output_dir)
    if not os.path.isdir(output_dir + "/destroyed/"):
        os.mkdir(output_dir + "/destroyed/")
    if not os.path.isdir(output_dir + "/major-damage/"):
        os.mkdir(output_dir + "/major-damage/")
    if not os.path.isdir(output_dir + "/minor-damage/"):
        os.mkdir(output_dir + "/minor-damage/")
    if not os.path.isdir(output_dir + "/no-damage/"):
        os.mkdir(output_dir + "/no-damage/")

    print("Looking for previously processed data file cache (" + cache_file + ") to save time...")
    if os.path.exists(cache_file):
        print("Cache file found, loading it...")
        cache_info = pickle.load(open(cache_file, "rb"))
    else:
        print("No cache file found, process all images (only needs to be done once)...")
        print("Identifying how many non-black pixels exist in each photo...")
        for (root, dirs, files) in os.walk(mask_image_location, topdown=True):
            for name in files:
                filename = os.path.join(root, name)
                non_empty_pixel_count = extract_non_empty_pixel_count(filename)
                cache_info.append([non_empty_pixel_count, filename])

                total = total + 1
                if total % 100 == 0:
                    print("  total processed count=" + str(total))

        print("  total processed count=" + str(total))
        print("Saving data for future use to " + cache_file + "...")
        pickle.dump(cache_info, open(cache_file, 'wb'))

    print("--- Statistics ---")
    items = pd.Series(x[0] for x in cache_info)
    print(items.describe())

    print("--- Filter out images ---")
    small_items = []
    filtered_items = []
    for item in cache_info:
        if item[0] == 0:
            print("ERROR: image does not contain any data->" + str(item[0]) + " --> " + item[1])
        small_items.append(item)

        # Filter out images of only a certain size and destruction type
        if (int(item[0]) >= filter_min_footprint) and (int(item[0]) <= filter_max_footprint) and \
                (filter_destruction_type in item[1]):
            filtered_items.append(item)

    # print("---Images with *really* small footprints---")
    # small_items_pd = pd.Series(x[0] for x in small_items)
    # print(small_items_pd.describe())

    print("-- Filtered dataset stats --")
    filtered_items_pd = pd.Series(x[0] for x in filtered_items)
    print(filtered_items_pd.describe())

    print("Copying files...")
    total_images = 0
    for filtered_item in filtered_items:
        outfile = re.sub(".*/maskedImage/", output_dir, filtered_item[1])
        print("Copying " + filtered_item[1] + " to " + outfile)
        shutil.copyfile(filtered_item[1], outfile)
        total_images = total_images + 1

    print("")
    print("Copied total_images=" + str(total_images) + " to " + output_dir)
    print("")
    print("Done.")
