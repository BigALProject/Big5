# Big5

* Team Members:
  - John Wagener
  - Kwame Sefah
  - Gavin Pedersen 
  - Vong Vang
  - Zach Aaberg

# Overview
This project is for SEIS 764 - Artificial Intelligence.

The goal of the project is to use satelite images pre and post natural disaster to automatically classify the damage done to a structure.

we took the images from https://xview2.org/dataset and pre-processed them by applying a mask to the images to isolate the houses.
Then we fed those images into a variety of AI algorithms including CNN, VG16, Graph, and Google.

General process (see user manual for complete details):
- Download bulk images from https://xview2.org/dataset
-- Original images from https://xview2.org/download-links.  You may need to sign up to get access.  THis contains the raw images.
    Download training set (images + labels)
    SHA1: 8298416c6f2c3bff28f6df55ffe7ff4a22bfc457
    Download additional Tier3 training data (images + labels)
    SHA1: 5bf6aaf8a71980b633fb4661776a99a200891de5
    Download test set (images)
    SHA1: fedae3b00b7ce47430b4850768e34aa449f0241d
- Transform images by using mask and filtering out each house.
- Pass images into AI.

