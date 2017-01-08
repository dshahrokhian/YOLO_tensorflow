# -*- coding: utf-8 -*-
"""
==========================================
Operations required for training a ConvNet with the VOC dataset.
==========================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import voc_utils as voc
import math
import numpy as np

grid_size = 7

def get_training_data(img_filename):
    annotation = voc.load_annotation(img_filename)
    
    img_width = int(annotation.find("width").text)
    img_height = int(annotation.find("height").text)
    
    objects = annotation.find_all("object")
    
    centers = _getXYWHC(objects, img_width, img_height)
    
def _getXYWHC(objects, img_width, img_height):
    '''
    Return a (grid_size)x(grid_size) grid with the center, width and height of the objects
    '''
    grid = [[None for x in range(grid_size)] for x in range(grid_size)]
    
    for obj in objects:
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        xmax = int(bbox.find("xmax").text)
        ymin = int(bbox.find("ymin").text)
        ymax = int(bbox.find("ymax").text)
        obj_class = obj.find("name").text
        
        width = xmax - xmin
        height = ymax - ymin
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
                
        cell_x, cell_y = getCell([center_x,center_y], img_width, img_height)
        grid[cell_x][cell_y] = [center_x, center_y, width, height, obj_class]
    print(grid)
    return grid

def _getP(self):
    pass

def getCell(point, width, height):
    '''
    Determines where a point falls within the (grid_size)x(grid_size) grid 
    '''
    row = math.floor(point[0] / width * grid_size)
    col = math.floor(point[1] / height * grid_size)

    return [row,col]

if __name__ == '__main__':
    get_training_data("2007_000027")