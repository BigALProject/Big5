import os
import pickle

import cv2
import numpy as np
import pandas as pd

mask_image_location = "../../data/maskedImage/"


def extract_non_empty_pixel_count(filename):
    image = cv2.imread(filename, 0)
    count = cv2.countNonZero(image)
    return count


if __name__ == "__main__":

    total = 0
    non_empty_counts = []

    print("Looking for previously processed data to save time...")
    if os.path.exists("non_empty_counts"):
        print("Cache file found, loading it...")
        non_empty_counts = pickle.load(open("non_empty_counts", "rb"))
    else:
        print("No cache file found, process all images...")
        print("Identifying how many non-black images exist in each photo...")
        for (root, dirs, files) in os.walk(mask_image_location, topdown=True):
            for name in files:
                filename = os.path.join(root, name)
                non_empty_pixel_count = extract_non_empty_pixel_count(filename)
                non_empty_counts.append([non_empty_pixel_count, filename])

                total = total + 1
                if total % 100 == 0:
                    print("  total processed count=" + str(total))

        print("  total processed count=" + str(total))
        print("Saving data for future use to non_empty_counts.txt")
        pickle.dump(non_empty_counts, open('non_empty_counts', 'wb'))

    print("---Statistics---")
    items = pd.Series(x[0] for x in non_empty_counts)
    print(items.describe())

    print("---Images with small footprints---")
    small_items = []
    large_items = []
    for item in non_empty_counts:
        if int(item[0]) < 50:
            if "no-damage" not in str(item[1]):
                #print("WARNING: " + str(item[0]) + " --> " + item[1])
                if item[0] == 0:
                    print("ERROR: " + str(item[0]) + " --> " + item[1])
                small_items.append(item)

        if int(item[0]) > 3000:
            large_items.append(item)

    print("---Images with *really* small footprints---")
    small_items_pd = pd.Series(x[0] for x in small_items)
    print(small_items_pd.describe())

    print("-- Large footprint --")
    large_items_pd = pd.Series(x[0] for x in large_items)
    print(large_items_pd.describe())
    for large_item in large_items:
        print(large_item[1])
        outfile = large_item[1].replace("maskedImage", "sample_training")
        print(outfile)

