import os
import json
import glob
import argparse
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

COCO_classes_ignore = [
 #'airplane',
 'apple',
 'backpack',
 'banana',
 'baseball bat',
 'baseball glove',
 'bear',
 'bed',
 'bench',
 #'bicycle',
 #'bird',
 #'boat',
 'book',
 'bottle',
 'bowl',
 'broccoli',
 #'bus',
 'cake',
 #'car',
 'carrot',
 #'cat',
 'cell phone',
 'chair',
 'clock',
 'couch',
 'cow',
 'cup',
 'dining table',
 #'dog',
 'donut',
 'elephant',
 'fire hydrant',
 'fork',
 'frisbee',
 'giraffe',
 'hair drier',
 'handbag',
 #'horse',
 'hot dog',
 'keyboard',
 'kite',
 'knife',
 'laptop',
 'microwave',
 #'motorcycle',
 'mouse',
 'orange',
 'oven',
 'parking meter',
 #'person',
 'pizza',
 'potted plant',
 'refrigerator',
 'remote',
 'sandwich',
 'scissors',
 'sheep',
 'sink',
 'skateboard',
 'skis',
 'snowboard',
 'spoon',
 'sports ball',
 #'stop sign',
 'suitcase',
 'surfboard',
 'teddy bear',
 'tennis racket',
 'tie',
 'toaster',
 'toilet',
 'toothbrush',
 #'traffic light',
 #'train',
 #'truck',
 'tv',
 'umbrella',
 'vase',
 'wine glass',
 'zebra']

coco2017toSTS = {
    "stop sign": "STOP",
    "motorcycle": "motorbike",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--json", type=str, required=True,
                        help='path to json file with annotation')
    parser.add_argument("--img-dir", type=str, required=True,
                        help='path to directory with relevant images')
    parser.add_argument("--names", type=str, required=True,
                        help='path to .names file in Darknet format')
    parser.add_argument("--num-classes", type=int, required=True,
                        help="number of unique classes")
    args = parser.parse_args()
    
    # Read JSON file
    path_imglab_json = args.json#"../../../Datasets/Swedish_Traffic_Signs/EmilyImages3/imx424-05-08_ground_truth/annotation/250_coco_imglab-2.json"
    with open(path_imglab_json) as f:
        data = json.load(f)
    print(f"Annotation file '{path_imglab_json}' loaded")
        
    # Encoder, Decoder
    img_to_idx = {img["file_name"]:img["id"] for img in data["images"]}
    img_to_size = {img["file_name"]:(img["width"], img["height"]) for img in data["images"]}
    idx_to_img = {img_to_idx[img]:img for img in img_to_idx}

    metadata = data["categories"]
    idx_to_label = {category["id"]: category["name"] for category in metadata}

    # Read all images in path
    path_imgs = args.img_dir#"../../../Datasets/Swedish_Traffic_Signs/EmilyImages3/imx424-05-08_ground_truth/"
    img_paths = glob.glob(path_imgs + "/*.jpg") + glob.glob(path_imgs + "/*.png")
    imgs = {os.path.basename(img_path):img_path for img_path in img_paths}

    # Aggregate all metadata into Python dictionary
    annot_dict = dict()
    for bbox in data["annotations"]:
        # imglab, image-id
        img_id = bbox['image_id']
        img = idx_to_img[img_id]
        
        # imglab, label-id
        annot_id = bbox["category_id"]
        label = idx_to_label[annot_id]
        
        # save bounding box into dictionary
        if annot_dict.get(img) is None:
            annot_dict[img] = [{
                "label": label,
                "bbox": bbox["bbox"]
            }]
        else:
            annot_dict[img].append({
                "label": label,
                "bbox": bbox["bbox"]
            })
            
    # Read .names file and create an class-to-id encoder
    with open(args.names) as f:
        lines = f.readlines()
        idx_to_class = [line.replace("\n", "") for line in lines]
        class_to_idx = {label: idx for idx, label in enumerate(idx_to_class)}
            
    # Loop through all images and save them into darknet format
    parent_dir = "darknet_format"
    Path(os.path.join(parent_dir, "images")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(parent_dir, "labels")).mkdir(parents=True, exist_ok=True)
    for file_name in imgs:
        img_path = imgs[file_name]
        
        # Copy image to this directory
        src = img_path
        dst = os.path.join(parent_dir, "images", file_name)
        shutil.copyfile(src, dst)
        
        # Write to a new .txt file in this directory
        if annot_dict.get(file_name) is None:
            print(f"Couldn't find {file_name} in annotation file.")
        else:
            bboxes = annot_dict[file_name]
        txt_name = file_name.replace(".jpg", ".txt").replace(".png", ".txt")
        size = img_to_size[file_name]
        
        with open(os.path.join(parent_dir, "labels", txt_name), "w") as f:
            for bbox in bboxes:
                label = bbox["label"]
                if label in COCO_classes_ignore:
                    pass
                else:
                    top_left_x = bbox["bbox"][0]
                    top_left_y = bbox["bbox"][1]
                    height = bbox["bbox"][3]
                    width = bbox["bbox"][2]

                    center_x = (top_left_x + width/2)
                    center_y = (top_left_y + height/2)
                    width /= size[0]
                    height /= size[1]
                    
                    if center_x < 0:
                        center_x = 0
                    elif center_x > size[0]:
                        center_x = size[0]
                    if center_y < 0:
                        center_y = 0
                    elif center_y > size[1]:
                        center_y = size[1]
                    
                    center_x /= size[0]
                    center_y /= size[1]
                    
                    
                    # Hard coded
                    #if label == "bike":
                    #    label = "bicycle"
                    #if label == "airplane":
                    #    label = "aeroplane"
                    #if label == "motorcycle":
                    #    label = "motorbike"
                    if coco2017toSTS.get(label) is not None:
                        label = coco2017toSTS[label]
                    
                    if class_to_idx.get(label) is None:
                        class_id = class_to_idx.get(label.upper()) 
                    else:
                        class_id = class_to_idx.get(label)
                    
                    #print(label)
                    if class_id is None:
                        print(f"'{label}' was not found in .names file.")
                        #print("Setting to default value: 0")
                        class_id = 0 # Or else will cause error
                    else:
                        
                        f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
                
    path = os.path.join(parent_dir, "images").replace("\\", "/")
    img_paths = glob.glob(path + "/*.jpg") + glob.glob(path + "/*.png")
    #print(img_paths)
    with open(os.path.join(parent_dir, "image_paths.txt"), "w") as f:
        for img_path in img_paths:
            img_path = img_path.replace("/", "\\")
            f.write(img_path + "\n")
            
    shutil.copyfile(args.names, f"{parent_dir}/labels.names")
    with open(os.path.join(parent_dir, "metadata.data"), "w") as f:
        f.write(f"classes={args.num_classes}\n")
        f.write(f"train={parent_dir}/image_paths.txt\n")
        f.write(f"valid={parent_dir}/image_paths.txt\n")
        f.write(f"names={parent_dir}/labels.names\n")

    
    
