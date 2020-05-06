# About this repository
This repository contains utility functions for visualization, testing and data conversion to Darknet format (YOLOv3 training) for Linköpings Traffic Sign dataset.

Most of the functions are hard-coded specifically for this dataset, but some of them could used in a more general sense.

# Requirements
 - opencv 4.1.2
 - pillow 6.2.0
 - matplotlib 3.1.1
 - seaborn 0.10.0
 - pandas 1.0.3

# Usage
Follow the following notebook to se examples of usage

<TODO: INSERT LINK HERE>

# Functions
In the following segment, useful are listed:

`visualization.py`
- Display an image with bounding boxes given data in Darknet format
  - `imshow_darknet(jpg_path: str, txt_path: str, names_path: str, figsize: Tuple[int,int] = (10, 10)) -> None`
  - Example usage: `$ python3 -c "from visualization import imshow_darknet; imshow_darknet('image/img05.jpg', 'labels/img05.txt')"`
                   
- Display two horizontal histograms given two lists
 - `two_stacked_horizontal_histogram(first_arr: List[str], second_arr: List[str], title: str = "Distribution of classes with the mirrored images", xlabel: str = "Frequency", ylabel: str = "Classes", legend: Tuple[str,str] = ('Original data', 'Mirrored')) -> None`
- Display a horizontal histogram given a list
 - `horizontal_histogram_counts(data: pd.Series, title: str, figsize: Tuple[int,int] = (15, 15)) -> None`

`tests.py`
- `test_darknet_txt_paths(path_to_samples_list: str)`
