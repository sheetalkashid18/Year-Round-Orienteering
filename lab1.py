"""
Author : SHEETAL SANTOSH KASHID
Subject : CSCI 630
LAB 1
"""

from PIL import Image
import numpy as np
import os, sys
import math
from queue import PriorityQueue
import queue

f = open(sys.argv[2], 'r')
elevations = []
elevations = np.array([line.split() for line in f])
rows, columns = np.shape(elevations)

elevations = np.delete(elevations, np.s_[columns - 5: columns + 1], 1)
elevations = elevations.transpose()
#print(elevations.shape)

#print(type(elevations))

#print(elevations)

terrain = Image.open(sys.argv[1])

#print(terrain.getpixel(320, 240))

rgbterrain = terrain.convert('RGB')

row, col = rgbterrain.size

f = open(sys.argv[3], 'r')
events = []
events = np.array([line.split() for line in f])

season = sys.argv[4]

fall = False
winter = False
summer = False
spring = False

if season == 'fall':
    fall = True
elif season == 'winter':
    winter = True
elif season == 'summer':
    summer = True
elif season == 'spring':
    spring = True

outputfile = sys.argv[5]


class Node:
    """
    To store the various values of a given pixel node
    """
    __slots__ = 'x', 'y', 'elevation', 'g', 'h', 'f', 'parent'

    def __init__(self, x, y, elevation, parent = None, h = 0, f = 0, g = 0):
        self.x = x
        self.y = y
        self.elevation = elevation
        self.g = g
        self.h = h
        self.f = f
        self.parent = parent

    # Sort nodes
    def __lt__(self, other):
        return self.elevation < other.elevation

    def __eq__(self, other):

        return (self.x, self.y) == (other.x, other.y)


def Astar(i, j):
    """
    A* search algorithm
    :param i: index of events for start
    :param j: index of events for destination
    :return: Path of traversal
    """

    closed = []

    visited = []

    opened = PriorityQueue()

    start = Node(int(events[i][0]), int(events[i][1]), float(elevations[int(events[i][0])][int(events[i][1])]))

    end = Node(int(events[j][0]), int(events[j][1]), float(elevations[int(events[j][0])][int(events[j][1])]))

    visited.append(start)

    opened.put((0, start))

    while opened.qsize() > 0:
        current = opened.get()[1]

        #print(current.x, current.y)

        if current in closed:
            continue

        closed.append(current)

        if current == end:
            path = []
            while current != start:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]

        x_coord = current.x
        y_coord = current.y

        next = [(x_coord-1, y_coord), (x_coord+1, y_coord), (x_coord, y_coord-1), (x_coord, y_coord+1)]

        for each in next:

            x, y = each

            if x >= row or x < 0 or y >= col or y < 0:
                continue

            terraintype = rgbterrain.getpixel(each)

            speedscale = calculatespeed(terraintype)

            if speedscale == -1:
                continue

            if fall:
                neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                for neighbour in neighbours:
                    neighbourterraintype = rgbterrain.getpixel(neighbour)
                    if neighbourterraintype == (255,255,255):
                        speedscale = speedscale * 1.4


            aNode = Node(x, y, elevations[x][y], current)
            aNode.g = distance(start.x, start.y, start.elevation, aNode.x, aNode.y, aNode.elevation)
            aNode.h = distance(end.x, end.y, end.elevation, aNode.x, aNode.y, aNode.elevation) * speedscale
            aNode.f = aNode.g + aNode.h

            if add(visited, aNode):
                opened.put((aNode.f, aNode))
                visited.append(aNode)

    return None

def add(visited, aNode):
    """
    Checks if a node should be added to visited
    :param visited:
    :param aNode:
    :return:
    """
    for eachnode in visited:
        if aNode == eachnode and aNode.f >= eachnode.f:
            return False
    return True


def distance(x1, y1, z1, x2, y2, z2):
    """
    Calculates 3d distance
    :return: 3d distance
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (float(z2) - float(z1))**2)


def calculatespeed(t):
    """
    Calculates scale factor wrt terrain
    :param t: terrain type
    :return: scale factor
    """

    if t == (248, 148, 18):
        return 0.7

    elif t == (71, 51, 3):
        return 0.4

    elif t == (0, 0, 0):
        return 0.6

    elif t == (255,255,255):
        return 0.8

    elif t == (2,208,60):
        return 0.9

    elif t == (2,136,40):
        return 1

    elif t == (255, 192, 0):
        return 1.1

    elif t == (0, 0, 255):
        return 1.5

    elif t == (135, 206, 250):
        return 0.2

    elif t == (210,105,30):
        return 1.6

    else:
        return -1


def findWaterEdges():
    """
    Edits walkable frozen ice
    :return: None
    """

    start = (0, 0)

    visited = [[False for i in range(col)] for j in range(row)]

    visitedqueue = queue.Queue()

    visitedqueue.put(start)


    while visitedqueue.qsize() > 0:

        current = visitedqueue.get()

        xn, yn = current

        if not visited[xn][yn]:
            visited[xn][yn] = True
            if rgbterrain.getpixel(current) == (0, 0, 255):
                for n in range(1, 8):
                    if not xn + n >= row:
                        if rgbterrain.getpixel((xn + n, yn)) != (0, 0, 255) and rgbterrain.getpixel((xn + n, yn)) != (135, 206, 250):
                            visited[xn + n][yn] = True
                            rgbterrain.putpixel(current, (135, 206, 250))
                            break
                        else:
                            visitedqueue.put((xn + n, yn))
                    if not xn - n < 0:
                        if rgbterrain.getpixel((xn - n, yn)) != (0, 0, 255) and rgbterrain.getpixel((xn - n, yn)) != (135, 206, 250):
                            visited[xn - n][yn] = True
                            rgbterrain.putpixel(current, (135, 206, 250))
                            break
                        else:
                            visitedqueue.put((xn - n, yn))
                    if not yn + n >= col:
                        if rgbterrain.getpixel((xn, yn + n)) != (0, 0, 255) and rgbterrain.getpixel((xn, yn + n)) != (135, 206, 250):
                            visited[xn][yn + n] = True
                            rgbterrain.putpixel(current, (135, 206, 250))
                            break
                        else:
                            visitedqueue.put((xn, yn + n))

                    if not yn - n < 0:
                        if rgbterrain.getpixel((xn, yn - n)) != (0, 0, 255) and rgbterrain.getpixel((xn, yn - n)) != (135, 206, 250):
                            visited[xn][yn - n] = True
                            rgbterrain.putpixel(current, (135, 206, 250))
                            break
                        else:
                            visitedqueue.put((xn, yn - n))
            else:
                neighbours = [(xn - 1, yn), (xn + 1, yn), (xn, yn - 1), (xn, yn + 1)]

                for neighbour in neighbours:

                    if neighbour[0] >= row or neighbour[0] < 0 or neighbour[1] >= col or neighbour[1] < 0:
                        continue
                    visitedqueue.put(neighbour)

            #print(neighbour)


def mudaccumulations():
    """
    edits mud accumulations
    :return: None
    """

    start = (0, 0)

    visited = [[False for i in range(col)] for j in range(row)]

    visitedqueue = queue.Queue()

    visitedqueue.put(start)

    while visitedqueue.qsize() > 0:

        current = visitedqueue.get()

        xn, yn = current

        if not visited[xn][yn]:

            visited[xn][yn] = True
            if rgbterrain.getpixel(current) == (0, 0, 255):
                for n in range(1, 16):
                    if not xn + n >= row:
                        if rgbterrain.getpixel((xn + n, yn)) != (0, 0, 255) and rgbterrain.getpixel((xn + n, yn)) != (
                        135, 206, 250) and rgbterrain.getpixel((xn + n, yn)) != (205,0,101) and abs(float(elevations[xn + n][yn]) - float(elevations[xn + n - 1][yn])) <= 1:
                            visited[xn + n][yn] = True
                            rgbterrain.putpixel((xn + n, yn), (210,105,30))
                        else:
                            visitedqueue.put((xn + n, yn))
                    if not xn - n < 0:
                        if rgbterrain.getpixel((xn - n, yn)) != (0, 0, 255) and rgbterrain.getpixel((xn - n, yn)) != (
                        135, 206, 250) and rgbterrain.getpixel((xn - n, yn)) != (205,0,101) and abs(float(elevations[xn - n][yn]) - float(elevations[xn - n + 1][yn])) <= 1:
                            visited[xn - n][yn] = True
                            rgbterrain.putpixel((xn - n, yn), (210,105,30))
                        else:
                            visitedqueue.put((xn - n, yn))
                    if not yn + n >= col:
                        if rgbterrain.getpixel((xn, yn + n)) != (0, 0, 255) and rgbterrain.getpixel((xn, yn + n)) != (
                        135, 206, 250) and rgbterrain.getpixel((xn, yn + n)) != (205,0,101) and abs(float(elevations[xn][yn + n]) - float(elevations[xn][yn + n - 1])) <= 1:
                            visited[xn][yn + n] = True
                            rgbterrain.putpixel((xn, yn + n), (210,105,30))
                        else:
                            visitedqueue.put((xn, yn + n))

                    if not yn - n < 0:
                        if rgbterrain.getpixel((xn, yn - n)) != (0, 0, 255) and rgbterrain.getpixel((xn, yn - n)) != (
                        135, 206, 250) and rgbterrain.getpixel((xn, yn - n)) != (205,0,101) and abs(float(elevations[xn][yn - n]) - float(elevations[xn][yn - n + 1])) <= 1:
                            visited[xn][yn - n] = True
                            rgbterrain.putpixel((xn, yn - n), (210,105,30))
                        else:
                            visitedqueue.put((xn, yn - n))
            else:
                neighbours = [(xn - 1, yn), (xn + 1, yn), (xn, yn - 1), (xn, yn + 1)]

                for neighbour in neighbours:

                    if neighbour[0] >= row or neighbour[0] < 0 or neighbour[1] >= col or neighbour[1] < 0:
                        continue
                    visitedqueue.put(neighbour)

if winter:
    findWaterEdges()
    #rgbterrain.save("water.png")

if spring:
    mudaccumulations()

cost = 0

for i in range(len(events)):
    if i < len(events) - 1:
        path = Astar(i, i+1)
        #print(path)
        if events[i][0] == path[0][0]:
            cost = cost + 10.29
        else:
            cost = cost + 7.55
        for j in range(len(path) - 1):
            if path[j][0] == path[j + 1][0]:
                cost = cost + 10.29
            else:
                cost = cost + 7.55
            rgbterrain.putpixel((path[j][0], path[j][1]), (255, 0, 0))
        rgbterrain.save(outputfile)


print(str(cost))












