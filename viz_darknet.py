import argparse

from utils.visualization import imshow_darknet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True,
                        help="Path to image")
    parser.add_argument("--annot", type=str, required=True,
                        help="Path to .txt annotation")
    parser.add_argument("--names", type=str, required=True,
                        help="Path to .names file")
    args = parser.parse_args()
    imshow_darknet(jpg_path = args.img, 
                   txt_path = args.annot, 
                   names_path = args.names, 
                   figsize = (10, 10))