import copy
import os
from PIL import Image

with open("../../dataset/new_train.txt", "r") as f:
    for line in f.readlines():
        orgin_line = copy.copy(line)
        line = line.strip()
        line = line.replace("hy-tmp/", "")
        for img in os.listdir(os.path.join(line, "rgb")):
            name = img.split(".")[0]
            tile = img.split(".")[1]
            if tile in ['png', 'jpg', 'jpeg']:
                try:
                    Image.open(os.path.join(line, 'rgb', img)).convert("RGB")
                except OSError as e:
                    os.remove(os.path.join(line, "rgb", img))
                    os.remove(os.path.join(line, "mask_visib", name + "_000000.png"))
                    os.remove(os.path.join(line, "axis_masks", name+".png"))
                    print(f"{line}/rgb/{img} is error opened....")
