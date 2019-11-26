import cv2
import os
import pandas as pd
import psycopg2 as pg2 
from shutil import copyfile

class ImageChipper():
    def __init__(self, dbconnection):
        # setting up the database connections and class variables
        # Connecting to the output database
        self.pginfo = dbconnection
        self.conn = pg2.connect(self.pginfo)

    def maskImage(self, source_img_path, mask_dir, img_mask_output, sql_query):
        with self.conn as c:
            df = pd.read_sql_query(sql_query, c)

            for row in df.iterrows():
                vals = row[1][3][6:].split(" ")
                uid = row[1][0]
                condition = row[1][1]
                image_id = row[1][2]
                image_name = image_id.split(".")[0]

                mask_name = f'{image_name}_{uid}.png' 
                output_name = f'{uid}_{condition}_{image_name}.png'

                full_image = os.path.join(source_img_path, image_id)
                mask_path = os.path.join(mask_dir, mask_name)
                chip_path = os.path.join(img_mask_output, condition, output_name)

                img = cv2.imread(full_image, cv2.IMREAD_COLOR)
                mask_in = cv2.imread(mask_path)
                try:
                    mask_img = cv2.cvtColor(mask_in, cv2.COLOR_BGR2GRAY)
                    img_mask = cv2.bitwise_and(img, img, mask=mask_img)

                    if condition == 'destroyed':
                        cv2.imwrite(chip_path, img_mask)
                    elif condition == 'major-damage':
                        cv2.imwrite(chip_path, img_mask)
                    elif condition == 'minor-damage':
                        cv2.imwrite(chip_path, img_mask)
                    elif condition == 'no-damage':
                        cv2.imwrite(chip_path, img_mask)
                    elif condition == 'un-classified':
                        cv2.imwrite(chip_path, img_mask)
                    else:
                        print("Error: Condition isn't valid.")
                except:
                    print('mask_img or another error')

        print(f"Finished {image_name}; {uid} - {condition}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def clip_maskImage(self, source_img_path, mask_dir, img_mask_output, sql_query, chip_size):        
        ''' Create chipped and masked images for training. Outputs a bunch of PNG image.'''   

        # Pixel size of image for axis. +- half to each centroid
        full_chip_size = chip_size

        chip_size = int(full_chip_size/2)

        with self.conn as c:
            df = pd.read_sql_query(sql_query, c)
            # print(df)

            for row in df.iterrows():

                vals = row[1][3][6:].split(" ")
                uid = row[1][0]
                condition = row[1][1]
                image_id = row[1][2]

                x = int(float(vals[0]))
                y = int(float(vals[1][:-1]))

                if x < full_chip_size:
                    pass

                elif y < full_chip_size:
                    pass
                else:
                    x_min = x - chip_size
                    x_max = x + chip_size
                    y_min = y - chip_size
                    y_max = y + chip_size

                    image_name = image_id.split(".")[0]

                    mask_name = f'{image_name}_{uid}.png' 
                    output_name = f'{uid}_{condition}_{image_name}.png'

                    full_image = os.path.join(source_img_path, image_id)
                    mask_path = os.path.join(mask_dir, mask_name)
                    chip_path = os.path.join(img_mask_output, condition, output_name)

                    img = cv2.imread(full_image, cv2.IMREAD_COLOR)
                    mask_in = cv2.imread(mask_path)
                    mask_img = cv2.cvtColor(mask_in, cv2.COLOR_BGR2GRAY)

                    img_mask = cv2.bitwise_and(img, img, mask=mask_img)

                    img_clip = img_mask[x_min:x_max, y_min:y_max]

                    if condition == 'destroyed':
                        cv2.imwrite(chip_path, img_clip)
                    elif condition == 'major-damage':
                        cv2.imwrite(chip_path, img_clip)
                    elif condition == 'minor-damage':
                        cv2.imwrite(chip_path, img_clip)
                    elif condition == 'no-damage':
                        cv2.imwrite(chip_path, img_clip)
                    elif condition == 'un-classified':
                        cv2.imwrite(chip_path, img_clip)
                    else:
                        print("Error: Condition isn't valid.")

                    print(f"Finished {image_name}; {uid} - {condition}")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    def renameMasks(self, directory, copy2newdir=False, masks_combo_output=''):
        masks_combo = r"{}".format(masks_combo_output)
        for dirs in os.listdir(directory):
            level2 = os.path.join(directory,dirs,'masks')
            for mask_file in os.listdir(level2):
                masking_path = os.path.join(level2,mask_file)
                output2_path = os.path.join(level2,mask_file.replace('pre', 'post'))
                try:
                    copyfile(masking_path, output2_path)
                except:
                    print("Mask already renamed and copied")
                combo_output_pre = os.path.join(masks_combo,mask_file)
                combo_output = os.path.join(masks_combo,mask_file.replace('pre', 'post'))
                if copy2newdir is True:
                    copyfile(output2_path, combo_output)
                    copyfile(masking_path, combo_output_pre)

if __name__ == "__main__":
    dbconn =  "host=ai-project-free.crx15vrrp87g.us-east-2.rds.amazonaws.com dbname=xview2 user=viewer password=viewer port=5431" # os.getenv('pg_spa') *** Using an environment variable instead.

    image_path_1 = r'C:\Users\Zerg\AI_Project\data\tier1_tier2\train\images'
    mask_input_path_1 = r'C:\Users\Zerg\AI_Project\data\tier1_tier2\masks_combo'

    image_path_2 = r'C:\Users\Zerg\AI_Project\data\tier3\train\images'
    mask_input_path_2 = r'C:\Users\Zerg\AI_Project\data\tier3\masks_combo'

    output_dir = r'C:\Users\Zerg\AI_Project\data\AI_TrainingImages\maskedImage'

    # Use one query or the other based on which dataset is being used
    # Tier 1 & 2 query:
    sql_query_1 = "select source_uid, condition, image_id, centroid from post_disaster_xy where image_id like 'guatemala-volcano%' or image_id like 'hurricane%' or image_id like 'mexico%' or image_id like 'midwest%' or image_id like 'palu%' or image_id like 'santa%' or image_id like 'socal%'"
            
    # Tier 3 query:
    sql_query_2 = "select source_uid, condition, image_id, centroid from post_disaster_xy where image_id like 'joplin-tornado%' or image_id like 'lower%' or image_id like 'moore-tornad%' or image_id like 'nepal%' or image_id like 'pinery-bush%' or image_id like 'portugal-wild%' or image_id like 'sunda-tsu%' or image_id like 'tuscaloosa%' or image_id like 'woolsey-fir%'"

    connection = ImageChipper(dbconn)
    
    # connection.clip_maskImage(image_path_1, mask_input_path_1, output_dir, sql_query_1, 224)

    connection.maskImage(image_path_1, mask_input_path_1, output_dir, sql_query_1)
    connection.maskImage(image_path_2, mask_input_path_2, output_dir, sql_query_2)

    # connection.renameMasks(r'C:\Users\Zerg\AI_Project\data\tier3\split')
    
