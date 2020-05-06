# About this repository
This repository contains utility functions for visualization, testing and data conversion to Darknet format (YOLOv3 training) for Linköpings Traffic Sign dataset.

Most of the functions are hard-coded specifically for this dataset, but some of them could used in a more general sense.

# Usage
Follow the following notebook to se examples of usage
<TODO: INSERT LINK HERE>

# Functions
In the following segment, useful are listed:

`visualization.py`
- `imshow_darknet(jpg_path: str, txt_path: str, names_path: str, figsize: Tuple[int,int] = (10, 10)) -> None`
                   
- `two_stacked_horizontal_histogram(first_arr: List[str], second_arr: List[str], title: str = "Distribution of classes with the mirrored images", xlabel: str = "Frequency", ylabel: str = "Classes", legend: Tuple[str,str] = ('Original data', 'Mirrored')) -> None`
- `horizontal_histogram_counts(data: pd.Series, title: str, figsize: Tuple[int,int] = (15, 15)) -> None`

`tests.py`
- `test_darknet_txt_paths(path_to_samples_list: str)`
