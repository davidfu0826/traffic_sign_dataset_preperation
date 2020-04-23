import os
from typing import Dict, List

def assert_data(annotation: str, img_dir: str) -> None:
    """Asserts that there is an image for every annotation for 
    Linköpings Traffic Sign dataset.

    Args:
      annotation: Annotation file (.txt)
      img_dir: Image directory
    """
    with open(annotation) as f:
        lines = f.readlines()

    imgs = [line[:25] for line in lines]
    imgs_ = os.listdir(img_dir)

    assert len(imgs_) == len(imgs)

    for img in imgs:
        assert img[-4:] == ".jpg"

    for img in imgs:
        assert img in imgs_
      
    print("All assertions have been passed smoothly.")
    
def test_types_values(dictionary: Dict, labels: List):
    """Asserts that all types and values are correct.
    """
    for img_path in dictionary:
        bboxes = dictionary[img_path]
        for bbox in bboxes:
    
            assert isinstance(bbox["signTypes"], str)
            assert isinstance(bbox["signBB"], tuple) 
            assert isinstance(bbox["signC"], tuple)

            assert bbox["signTypes"] in set(labels)
            
            for coord in bbox["signBB"]:
                assert coord >= 0
            for coord in bbox["signC"]:
                assert coord >= 0
    print("All assertions have been passed smoothly.")
    
def test_darknet_txt_paths(path_to_samples_list: str):
    """Asserts that all files listed the train-, valid- images for Darknet format dataset
    exists and their corresponding labels too.
    """
    with open(path_to_samples_list, "r") as f:
        data = f.readlines()
        for img_path in [path[:-1] for path in data]:
            assert os.path.isfile(img_path)
            assert os.path.isfile(img_path.replace("/images/", "/labels/").replace(".jpg", ".txt"))
    print(f"All files listed in '{path_to_samples_list}' exists.")