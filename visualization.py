from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd

def horizontal_histogram_counts(data: pd.Series, title: str, figsize: Tuple[int,int] = (15, 15)):
    """Displays a horizontal histogram over class frequencies.  

    Args:
        data: Pandas series of sample labels
        title: Title for the figure 
        figsize: Size of the figure (optional)
    """

    plt.figure(figsize=figsize)
    ax = sns.countplot(y=data, order=data.value_counts().index)

    plt.title(title)
    plt.xlabel('Frequency')

    for p in ax.patches:
        x = p.get_x() + p.get_width() + 0.3
        y = p.get_y() + p.get_height()/2 + 0.2
        ax.annotate(p.get_width(), (x, y))

    plt.show()
    
def plot_image_with_bbox(idx: int, annot_dict: Dict, figsize=(10,10)):
    """Displays an image with a bounding box from
    LIU Traffic sign dataset.
    """

    imgs = [img for img in annot_dict]

    # Get a sample
    img_path = imgs[idx]
    annots = annot_dict[img_path]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # This loops over all bounding boxes for this sample
    for annot in annots:
        label = annot["signTypes"]
        center_x, center_y = annot["signC"]
        center_x, center_y = int(center_x), int(center_y)
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = annot["signBB"]
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(img, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
        img = cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

    print(img.shape)
    plt.figure(figsize=figsize)
    plt.rcParams["axes.grid"] = False
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
def two_stacked_horizontal_histogram(
                        first_arr: List[str], second_arr: List[str],
                        title: str = "Distribution of classes with the mirrored images",
                        xlabel: str = "Frequency",
                        ylabel: str = "Classes",
                        legend: Tuple[str,str] = ('Original data', 'Mirrored')) -> None:
    """Plots two stacked histograms given two arrays
    """
    s1=pd.Series(first_arr, name="original")
    s2=pd.Series(second_arr, name="mirrored")

    df = pd.concat([s1, s2])

    plt.figure(figsize=(15, 15))

    total_counts = df.value_counts() 
    original_counts = s1.value_counts().reindex(total_counts.index)

    #Plot 1 - background - "total" (top) series
    ax = sns.barplot(y = total_counts.index, x = total_counts, color = "red", alpha=0.5)

    # Add values on top of the bar
    for p in ax.patches:
        x = p.get_x() + p.get_width() + 0.9
        y = p.get_y() + p.get_height()/2 + 0.2
        ax.annotate(int(p.get_width()), (x, y))

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(y = total_counts.index, x = original_counts, color = "#0000F3", alpha=0.5)

    bottom_plot.axes.set_title(title, fontsize=30)
    bottom_plot.set_xlabel(xlabel, fontsize=20)
    bottom_plot.set_ylabel(ylabel, fontsize=20)

    topbar = plt.Rectangle((0,0),1,1, fc="#F77", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1, fc='#965596',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], legend, loc='center right', ncol = 2, prop={'size':24})
    l.draw_frame(False)

    plt.xlabel('Frequency')
    
    
def imshow_darknet(jpg_path: str, 
                   txt_path: str, 
                   names_path: str, 
                   figsize: Tuple[int,int] = (10, 10)) -> None:
    """Displays images in a dataset in Darknet format.
    """
    with open(names_path, "r") as f:
        labels = f.readlines()
        labels = [label[:-1] for label in labels]

    idx_to_label = dict()
    for idx, label in enumerate(labels):
        idx_to_label[idx] = label

    with open(txt_path, "r") as f:
        bboxes = f.readlines()

    img = cv2.imread(jpg_path)
    height, width, _ = img.shape

    for data_raw in bboxes:
        data = data_raw[:-1].split(" ")
        label = idx_to_label[int(data[0])]

        #class x_center y_center width height
        center_x, center_y      = float(data[1])*width, float(data[2])*height
        bbox_width, bbox_height = float(data[3])*width, float(data[4])*height

        pt1 = (int(center_x + bbox_width/2), int(center_y + bbox_height/2))
        pt2 = (int(center_x - bbox_width/2), int(center_y - bbox_height/2))
        pt3 = (int(center_x), int(center_y)+10)

        img = cv2.putText(img, label, pt3, cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,255,0)) 
        img = cv2.rectangle(img, pt1, pt2, (0,255,0), 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img)