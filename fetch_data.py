import flickrapi
import json
import requests
import os
import sys
import io
from PIL import Image, ImageOps
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category", type=str, help="class to train")
    args = parser.parse_args()
    CATEGORY = args.category

print(CATEGORY)

image_size = (40, 40) # (width, height)
output_file_dir = "./data/"

with open("./settings.config", "rb") as settings_file:
    settings = json.load(settings_file)
    flickr = flickrapi.FlickrAPI(settings["KEY"], settings["SECRET"], format='parsed-json')

# NOTE: total must be divisible by 100
def get_images(tags, total):
    if total % 100 != 0:
        print "Total must be divisible by 100!"
        sys.exit()

    total_requests = total / 100

    to_return = []
    for page in xrange(1, 1 + total_requests):
        response = flickr.photos.search(text=tags, per_page=100, page=page, sort='relevance')
        for photo in response["photos"]["photo"]:
            url = "https://farm{}.staticflickr.com/{}/{}_{}.jpg".format(photo["farm"], photo["server"], photo["id"], photo["secret"])
            to_return.append(url)
    return to_return

for tag, filetag, filenum in [(CATEGORY, CATEGORY, 100), ('-' + CATEGORY, 'not' + CATEGORY, 100)]:
    # create dir
    cur_output_dir = os.path.join(output_file_dir, filetag)
    if not os.path.exists(cur_output_dir):
        os.makedirs(cur_output_dir)

    # get images
    i = 0
    for image in get_images(tag, filenum):
        # create the image
        stream = io.BytesIO(requests.get(image).content)
        img = Image.open(stream)

        # resize
        cropped_img = ImageOps.fit(img, image_size, method=Image.BICUBIC)

        # output
        cropped_img.save(os.path.join(cur_output_dir, str(i) + '.jpg'))

        i += 1
