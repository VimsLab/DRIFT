import numpy as np
import cv2

def geodesic_distance_transform(img, center):
    """return geodesic map from the point"""
    """ Unfinished"""
    img = img.astype(np.float32)
    image = img.copy();
    image[0,:] = -2;
    image[len(image) - 1,:]=-2
    image[:,0] = -2
    image[:, len(image[0]) - 1] = -2

    img_not_visit = image
    img_mark = img_not_visit>0

    img_tmp = image
    img_tmp[~img_mark] = -2
    img_tmp[img_mark] = -1
    indexes = np.where(image > 0)


    curr_point = center
    img_tmp[center] = 0
    curr_dist = 0

    while(len(np.where(img_mark == True)[0]) > 0):
        print (len(np.where(img_mark == True)[0]))
        indexes = np.where(img_tmp == curr_dist)
        print(curr_dist)
        # import pdb; pdb.set_trace()

        for i in range(len(indexes[0])):
            img_mark[indexes[0][i], indexes[1][i]] = False
            print ('Center----------------------------')
            print ([indexes[0][i], indexes[1][i]])
            # import pdb; pdb.set_trace()
            for x_coor in range(-1,2):

                for y_coor in range (-1,2):



                    if (img_mark[indexes[0][i] + x_coor, indexes[1][i] +y_coor] == True):
                        print ('Before----------------------------')
                        print ([indexes[0][i] + x_coor, indexes[1][i] +y_coor])
                        img_tmp[indexes[0][i] + x_coor, indexes[1][i] +y_coor] = curr_dist + 1
                        img_mark[indexes[0][i] + x_coor, indexes[1][i] +y_coor] = False

        curr_dist = curr_dist + 1
        # import pdb; pdb.set_trace()
        if len(indexes[0]) < 1 :
            break

    return img_tmp

def get_skeleton_endpoint(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6

    validEndpoint = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]];
    image = skeleton.copy();
    image[0,:] = 0;
    image[len(image) - 1,:]=0
    image[:,0] = 0
    image[:, len(image[0]) - 1] = 0
    intersections = list();
    endpoints = list();
    indexes = np.where(image>0);

    # import pdb; pdb.set_trace()
    for i in range(len(indexes[0])):
        neighbours = neighbour(indexes[0][i], indexes[1][i],image);
        valid = True;
        if neighbours in validEndpoint:
            endpoints.append((indexes[1][i], indexes[0][i]))

    # for x in range(1,len(image)-1):
    #     for y in range(1,len(image[x])-1):
    #         # If we have a white pixel
    #         if image[x][y] == 1:
    #             neighbours = neighbour(x,y,image);
    #             valid = True;
    #             if neighbours in validIntersection:
    #                 intersections.append((y,x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together

    # Remove duplicates
    return endpoints;

def neighbour(x,y,image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

def get_skeleton_intersection_and_endpoint(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];

    validEndpoint = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]];
    image = skeleton.copy();
    image[0,:] = 0;
    image[len(image) - 1,:]=0
    image[:,0] = 0
    image[:, len(image[0]) - 1] = 0
    intersections = list();
    endpoints = list();
    indexes = np.where(image>0);

    # import pdb; pdb.set_trace()
    for i in range(len(indexes[0])):
        neighbours = neighbour(indexes[0][i], indexes[1][i],image);
        valid = True;
        if neighbours in validIntersection:
            intersections.append((indexes[1][i],indexes[0][i]))
        if neighbours in validEndpoint:
            endpoints.append((indexes[1][i], indexes[0][i]))

    # for x in range(1,len(image)-1):
    #     for y in range(1,len(image[x])-1):
    #         # If we have a white pixel
    #         if image[x][y] == 1:
    #             neighbours = neighbour(x,y,image);
    #             valid = True;
    #             if neighbours in validIntersection:
    #                 intersections.append((y,x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections, endpoints;




def get_skeleton_intersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];

    validEndpoint = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]];
    image = skeleton.copy();
    image[0,:] = 0;
    image[len(image) - 1,:]=0
    image[:,0] = 0
    image[:, len(image[0]) - 1] = 0
    intersections = list();

    indexes = np.where(image>0);

    # import pdb; pdb.set_trace()
    for i in range(len(indexes[0])):
        neighbours = neighbour(indexes[0][i], indexes[1][i],image);
        valid = True;
        if neighbours.count(1) > 2:
            intersections.append((indexes[1][i],indexes[0][i]))

    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 5**2) and (point1 != point2):
                intersections.remove(point2);

    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 5**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections
