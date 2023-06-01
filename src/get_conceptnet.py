import os
import requests

def download_conceptnet():
    # Define the url and the file name
    url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
    file_name = "conceptnet-assertions-5.7.0.csv.gz"

    # Check if the file already exists
    if os.path.exists(file_name):
        print("File already exists. Skipping download.")
    else:
        # Download the file and save it
        print("Downloading file...")
        response = requests.get(url)
        with open(file_name, "wb") as f:
            f.write(response.content)
        print("Download completed.")


download_conceptnet()