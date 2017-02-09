import voc_utils as voc
import math

grid_size = 7
network_reshape = 448 # The convolutional neural network reduces the image size to 448x448

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
        center_x = float(xmin + xmax) / 2
        center_y = float(ymin + ymax) / 2

        #bbox = reshape([center_x,center_y,width,height],
        #               img_width, img_height, 
        #               network_reshape, network_reshape)
                
        cell_x, cell_y = getCell([center_x,center_y], img_width, img_height)

        if (grid[cell_x][cell_y] is None):
            grid[cell_x][cell_y] = [[center_x, center_y, width, height, obj_class]]
        else:
            grid[cell_x][cell_y] = [grid[cell_x][cell_y][0], [center_x, center_y, width, height, obj_class]]

    return grid

def getCell(point, width, height):
    '''
    Determines where a point falls within the (grid_size)x(grid_size) grid 
    '''
    col = int(math.floor(point[0] / width * (grid_size-1)))
    row = int(math.floor(point[1] / height * (grid_size-1)))

    return [row,col]
    
def reshape(bbox, original_width, original_height, new_width, new_height):
    
    w_ratio = new_width / original_width
    h_ratio = new_height / original_height
    
    return [bbox[0] * w_ratio, 
            bbox[1] * h_ratio, 
            bbox[2] * w_ratio, 
            bbox[3] * h_ratio]
    