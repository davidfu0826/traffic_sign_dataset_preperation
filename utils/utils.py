import os
from typing import Dict

from PIL import Image

def is_valid_decimal(string: str) -> bool:
    """Given a string e.g. "2.29" or "2.4a2",
    determine if you can convert it to a valid
    float.

    Example:
      "2.29" --> True
      "David" --> False
      "2.4a2" --> False
    """
    try:
        float(string)
    except ValueError:
        return False
    else:
        return True

def cutoff_letter(string: str) -> str:
    """Keeps the first numerical characters.

    For instance:
    Input:  478.4376a0
    Output: 478.4376

    Input:  318.3432b0
    Output: 318.3432
    """
    for idx, char in enumerate(string):
        if char.isalpha():
            return string[:idx]

def extract_data(filename: str, directory: str) -> Dict:
    """Extracts all the metadata from Traffic Sign Dataset
    into an dictionary (~JSON like structure).

    Notes:
      Image path and annotation separated by :
      Each bounding box is separated by ;

    Args:
      filename: Annotation file
      directory: Path to these images
    """
    with open(filename) as f:
        lines = f.readlines()

    # Split data by :
    annotations = [line.replace(" ", "").split(":") for line in lines]

    # Split data by ;
    for annotation in annotations:
        annotation[1] = annotation[1].split(";")

    # Loop for saving metadata into dictionary
    annot_dict = dict()
    for annotation in annotations:
        img = annotation[0]
        bbox_metadata = annotation[1]
        bbox = list()
        
        # Path to images
        img_path = os.path.join(directory, img)
        im = Image.open(img_path)
        width, height = im.size

        # Iterate over each bounding box
        for annot in bbox_metadata:
            
            if "MISC_SIGNS" == annot:
                signStatus = 'N/A'
                signTypes = "MISC_SIGNS"
                signPurpose = 'N/A'

                signBB = (-1, -1, -1, -1)
                signC = (-1, -1)
                signSize = 0
                aspectRatio = 0

                bbox.append({"signStatus": signStatus, 
                            "signTypes": signTypes, 
                            "signPurpose": signPurpose, 
                            "signBB": signBB, 
                            "signC": signC, 
                            "signSize": signSize, 
                            "aspectRatio": aspectRatio})
            elif "\n" in annot:
                pass
            else:
                data = annot.split(",")
              
                signStatus = data[0] # signStatus
                signTypes = data[6] # signTypes
                signPurpose = data[5] # PROHIBITORY, WARNING, OTHER, INFORMATION
                tl_x, tl_y, br_x, br_y = data[3], data[4], data[1], data[2]
                
                if is_valid_decimal(tl_x):
                    tl_x = float(tl_x)
                else:
                    tl_x = float(cutoff_letter(tl_x))

                if is_valid_decimal(tl_y):
                    tl_y = float(tl_y)
                else:
                    tl_y = float(cutoff_letter(tl_y))

                if is_valid_decimal(br_x):
                    br_x = float(br_x)
                else:
                    br_x = float(cutoff_letter(br_x))

                if is_valid_decimal(br_y):
                    br_y = float(br_y)
                else:
                    br_y = float(cutoff_letter(br_y))

                if tl_x < 0:
                    tl_x = 0
                elif tl_x > width:
                    tl_x = width
                                        
                if tl_y < 0:
                    tl_y = 0
                elif tl_y > height:
                    tl_y = height
                    
                if br_x < 0:
                    br_x = 0
                elif br_x > width:
                    br_x = width
                    
                if br_y < 0:
                    br_y = 0
                elif br_y > height:
                    br_y = height

                signBB = (tl_x, tl_y, br_x, br_y)
                signC = (br_x + tl_x)/2, (br_y + tl_y)/2
                signSize = (br_x - tl_x) * (br_y - tl_y)
                aspectRatio = (br_x - tl_x) / (br_y - tl_y)

                bbox.append({"signStatus": signStatus, 
                            "signTypes": signTypes, 
                            "signPurpose": signPurpose, 
                            "signBB": signBB, 
                            "signC": signC, 
                            "signSize": signSize, 
                            "aspectRatio": aspectRatio})
            
            
            annot_dict[img_path] = bbox
    return annot_dict

def read_annot(path: str) -> Dict:
    """Reads annotation files from Linköping Traffic Sign dataset
    
    Args:
        path: Path to annotation file (.txt)
    """
    with open(path) as f:
        lines = f.readlines()
        lines = [line.replace("\n", "").replace(" ", "").split(":") for line in lines]
        data = {line[0]: line[1].split(";") for line in lines}

    for img in data:
        bboxes = data[img]
        #print(bboxes)
        new_bboxes = list()
        for bbox in bboxes:
            if bbox == "MISC_SIGNS":
                pass
            else:
                if bbox == "":
                    pass
                else:
                    bbox_data = bbox.split(",")
                    bbox_coord = bbox_data[1:5]
                    label = bbox_data[6]
                    if label == "URDBL":
                        label = "OTHER"

                    for i, coord in enumerate(bbox_coord):
                        if is_valid_decimal(coord):
                            bbox_coord[i] = float(coord)
                        else:
                            bbox_coord[i] = float(cutoff_letter(coord))
               
                    new_bbox = {
                        "bbox": bbox_coord,
                        "label": label
                        }
                    new_bboxes.append(new_bbox)
        data[img] = new_bboxes
    return data
