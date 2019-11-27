import psycopg2 as pg2
import json
import os

class DataLoad():
    def __init__(self, dbconnection):
        # setting up the database connections and class variables
        # Connecting to the output database   
        self.pginfo = dbconnection 
        self.conn = pg2.connect(self.pginfo)

        # Create needed tables in PG DB
        # self.table_init()
      
    def table_init(self):        
        ''' Creates a blank table on PG for JSON to be written to.'''   
        with self.conn, self.conn.cursor() as c:
            print("\nInitializing PG DB...")
            c.execute("CREATE TABLE IF NOT EXISTS post_disaster_geo (id SERIAL PRIMARY KEY, source_uid text, condition text, disaster text, disaster_type text, cat_id text, image_id text, metadata text, wkt text)")
            c.execute("CREATE TABLE IF NOT EXISTS pre_disaster_geo (id SERIAL PRIMARY KEY, source_uid text, condition text, disaster text, disaster_type text, cat_id text, image_id text, metadata text, wkt text)")
            # c.execute("TRUNCATE TABLE post_disaster_xy")
            # c.execute("TRUNCATE TABLE pre_disaster_xy")

    def readJSON(self, input_file):

        with open(input_file) as f:

            if "pre_disaster.json" in input_file:
                line = json.load(f)
                lng_lat = line['features']['lng_lat']

                metadata = str(line["metadata"])
                print(f"Metadata: {metadata}\n")

                for feat in lng_lat:
                    ftype = feat['properties']['feature_type']
                    uid = feat['properties']['uid']
                    wkt = feat['wkt']

                    disaster = line["metadata"]["disaster"]
                    disaster_type = line["metadata"]["disaster_type"]
                    catalog_id = line["metadata"]["catalog_id"]
                    img_id = line["metadata"]["img_name"]

                    with self.conn, self.conn.cursor() as c:
                        c.execute(
                            "INSERT INTO pre_disaster_geo (source_uid, disaster, disaster_type, cat_id, image_id, metadata, wkt) \
                                VALUES (%s, %s, %s, %s, %s, %s, %s);", (uid, disaster, disaster_type, catalog_id, img_id, metadata, wkt))
                        # c.execute("select st_setsrid(a.wkt, 4326) from pre_disaster a;")
                        # c.execute("SELECT UpdateGeometrySRID('pre_disaster','wkt',0);")


            else:
                line = json.load(f)
                lng_lat = line['features']['lng_lat']

                metadata = str(line["metadata"])
                print(f"Metadata: {metadata}\n")

                for feat in lng_lat:
                    ftype = feat['properties']['feature_type']
                    uid = feat['properties']['uid']
                    condition = feat['properties']['subtype']
                    wkt = feat['wkt']

                    disaster = line["metadata"]["disaster"]
                    disaster_type = line["metadata"]["disaster_type"]
                    catalog_id = line["metadata"]["catalog_id"]
                    img_id = line["metadata"]["img_name"]

                    with self.conn, self.conn.cursor() as c:
                        c.execute(
                            "INSERT INTO post_disaster_geo (source_uid, condition, \
                                disaster, disaster_type, cat_id, image_id, \
                                    metadata, wkt) VALUES (%s, %s, %s, %s, %s,\
                                        %s, %s, %s);", (uid, condition,
                                            disaster, disaster_type, catalog_id,
                                            img_id, metadata, wkt))


if __name__ == "__main__":
    dbconn =  os.getenv('xview2_pg')

    connection = DataLoad(dbconn)

    input_directory = r"C:\Users\Zerg\AI_Project\data\tier3\train\label2"
    for file_name in os.listdir(input_directory):
        print(file_name)
        connection.readJSON(os.path.join(input_directory, file_name))
