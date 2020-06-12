# About this repository
This repository contains utility functions for visualization, testing and data conversion to Darknet format (YOLOv3 training) for Linköpings Traffic Sign dataset. The reason why this repository came to existance was because the code looked like a spaghetti mess until I decided to encapsulate all code into functions for cleaner code.

Most of the functions are hard-coded specifically for this dataset, but some of them could used in a more general sense.

**Updates will probably come!**

# Requirements
Not tested with other versions, but everything should work for an forseeable future.
 - opencv 4.1.2
 - pillow 6.2.0
 - matplotlib 3.1.1
 - seaborn 0.10.0
 - pandas 1.0.3

# Usage
Follow the following notebook to se examples of usage

<TODO: INSERT LINK HERE>

# Functions
In the following segment, very useful functions are listed:

## CLI commands
#### Plot image with bounding boxes in Darknet format with `viz_darknet.py`
- `python3 viz_darknet.py --img <img path> --annot <txt path> --names <path label file>`
- Example CLI usage: `$ python3 viz_darknet.py --img image/img05.jpg --annot labels/img05.txt --names something/metadata.names`
#### Convert COCO JSON to Darknet format with `coco2darknet.py`

    parser.add_argument("--names", type=str, required=True,
                        help='path to .names file in Darknet format')
    parser.add_argument("--num-classes", type=int, required=True,
                        help="number of unique classes")
- `python3 coco2darknet.py --json <json path> --img-dir <img dir path> --names <path label file> --num-classes <int>` 
- Example CLI usage: `$ python3 coco2darknet.py --json data/coco_imglab.json --img-dir data/images --names data/traffic_signs.names --num-classes 33`

## Functions that can be used in python script/notebooks
`visualization.py`
- **Stacked horizontal histograms (stacking two histograms)**
  - `two_stacked_horizontal_histogram(first_arr: List[str], second_arr: List[str], title: str = "Distribution of classes with the mirrored images", xlabel: str = "Frequency", ylabel: str = "Classes", legend: Tuple[str,str] = ('Original data', 'Mirrored')) -> None`
  
- **Horizontal histogram**
  - `horizontal_histogram_counts(data: pd.Series, title: str, figsize: Tuple[int,int] = (15, 15)) -> None`

- **Plot image + bounding boxes in Darknet format**
  - `imshow_darknet(jpg_path: str, txt_path: str, names_path: str, figsize: Tuple[int,int] = (10, 10)) -> None`

`tests.py`
- Check there is annotation (/labels/\*\*.txt) for every image (/images/\*\*.jpg) in Darknet format
  - `test_darknet_txt_paths(path_to_samples_list: str)`

The rest of functions are hard-coded and too specific to the dataset, therefore they won't be described here (if you insist, look in the notebook).
