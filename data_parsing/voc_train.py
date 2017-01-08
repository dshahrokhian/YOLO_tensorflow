# -*- coding: utf-8 -*-
"""
==========================================
Utils required for training a ConvNet with the VOC dataset.
==========================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import voc_utils as voc
import math

grid_size = 7

def get_training_data(img_filename):
    annotation = voc.load_annotation(img_filename)
    
    img_width = int(annotation.find("width").text)
    img_height = int(annotation.find("height").text)
    
    objects = annotation.find_all("object")
    
    return _getXYWHC(objects, img_width, img_height)
    
def _getXYWHC(objects, img_width, img_height):
    '''
    Return a (grid_size)x(grid_size) grid with the center, width, height and class of the objects
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
        
        if (grid[cell_x][cell_y] is None):
            grid[cell_x][cell_y] = [center_x, center_y, width, height, obj_class]
        else:
            grid[cell_x][cell_y] = [grid[cell_x][cell_y], [center_x, center_y, width, height, obj_class]]

    return grid

def getCell(point, width, height):
    '''
    Determines where a point falls within the (grid_size)x(grid_size) grid 
    '''
    row = math.floor(point[0] / width * grid_size)
    col = math.floor(point[1] / height * grid_size)

    return [row,col]
