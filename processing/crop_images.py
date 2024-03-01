"""
Meant to be used to fetch and save all the crops from images using Yolov8 annotations in .txt xywh (relative).
Usage python tools/classification/crop_images.py -p <PATH_TO_FOLDER/> -f _labels -o ./data/crops/ -c 3
where <PATH_TO_FOLDER> should be the path to the folder which contains the images and _labels subfolders
"""

import click
import torch
import cv2
import numpy as np
import logging

from typing import Tuple,Optional,List
from multiprocessing import Pool
from pathlib import Path
from functools import partial


DEFAULT_NUM_WORKERS=2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def xywh2xyxy(box:list,image_width:int,image_height:int):
    """
    converts annotations from yolov8 in xywh (Relative)
    to absolute positions
    """
    cls_id=int(box[0])
    x_center=box[1]
    y_center=box[2]
    width=box[3]
    height=box[4]
    x_min = int((x_center - (width / 2)) * image_width)
    y_min = int((y_center - (height / 2)) * image_height)
    x_max = int((x_center + (width / 2)) * image_width)
    y_max = int((y_center + (height / 2)) * image_height)
    return [cls_id,x_min,y_min,x_max,y_max]

def txt2box(txt_path:str,filter_cls: Optional[str]):
    """Reads and parse bounding boxes in a .txt with Yolov8 format

    Args:
        txt_path (str): path to a txt annotations file
        filter_cls Optional[str]: class id to get from annotations, use None if all the classes
        are required.

    Returns:
        List[float]: List of floats with the cls as the first position, x,y,w,h in the remaining
        positions
    """
    with open(txt_path,'r') as f:
        labels=[list(map(float,i.replace("\n","").split())) for i in f.readlines()]
    return labels if filter_cls is None else [l for l in labels if int(l[0])==filter_cls]

def get_image_crops(image:np.ndarray,boxes:List):
    """
    image read with cv2 (h,w)
    boxes: list of bboxes in absolute format (x_min,y_min,x_max,y_max)
    """
    crops=[]
    for box in boxes:
        x_i,y_i,x_f,y_f=box[1:]
        img=image[y_i:y_f, x_i:x_f]
        crops.append(img)
    return crops

def txt2crops(root_path:str,out_path:str,filter_cls,txt_path:Path):
    boxes=txt2box(str(txt_path),filter_cls=filter_cls)

    try:
        img_name=txt_path.name.replace('.txt','.jpg')
        img_pth=str(Path(root_path)/f"images/{img_name}")
        img=cv2.imread(img_pth)
        img_h,img_w,_=img.shape
        boxes_abs=[xywh2xyxy(b,img_w,img_h) for b in boxes]
        crops=get_image_crops(img,boxes_abs)
        for idx,cp in enumerate(crops,1):
            cp_outpath=str(Path(out_path)/f'{idx}-{img_name}')
            cv2.imwrite(cp_outpath,cp)
            logger.info(f"saved crop to {cp_outpath}")
    except Exception as e:
        logger.info(f"Could not process {str(txt_path)}:{e}")    
        
@click.command()
@click.option('-p', '--root_path', type=click.Path(exists=True), required=True)
@click.option('-f','--folder_name_labels',type=str,required=True, default="_labels")
@click.option('-o', '--output_folder', type=click.Path(exists=False), help="Output folder where to save the crops",required=True)
@click.option('-c','--filter_cls',type=int,required=False,default=None,help="Default cls id to include in the crops")
def main(root_path,folder_name_labels,output_folder,filter_cls):
    #make sure output dir exists
    Path(output_folder).mkdir(exist_ok=True,parents=True)

    label_paths=Path(root_path)/folder_name_labels
    label_paths=list(label_paths.glob("*.txt"))

    #partially defines the method and first arguments
    func=partial(txt2crops,root_path,output_folder,filter_cls)
    
    with Pool(DEFAULT_NUM_WORKERS) as pool:
        image_crops = pool.map(func,label_paths)

if __name__=="__main__":
    main()