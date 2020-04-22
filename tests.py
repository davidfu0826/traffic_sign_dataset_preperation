import os

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
    
def test_types_values(dictionary, labels):
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