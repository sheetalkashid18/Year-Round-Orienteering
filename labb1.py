from PIL import Image
import numpy as np
import os, sys
import math
from queue import PriorityQueue
import queue

f = open('mpp.txt', 'r')
elevations = []
elevations = np.array([line.split() for line in f])
rows, columns = np.shape(elevations)

elevations = np.delete(elevations, np.s_[columns - 5: columns + 1], 1)
elevations = elevations.transpose()
print(elevations.shape)

#print(type(elevations))

#print(elevations)

terrain = Image.open("terrain.png")

#print(terrain.getpixel(320, 240))

rgbterrain = terrain.convert('RGB')

row, col = rgbterrain.size

f = open('brown.txt', 'r')
events = []
events = np.array([line.split() for line in f])

fall = False
winter = True
spring = False
summer = True


class Node:
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

    closed = []

    visited = []

    opened = PriorityQueue()

    start = Node(int(events[i][0]), int(events[i][1]), float(elevations[int(events[i][0])][int(events[i][1])]))

    end = Node(int(events[j][0]), int(events[j][1]), float(elevations[int(events[j][0])][int(events[j][1])]))

    visited.append(start)

    opened.put((0, start))

    while opened.qsize() > 0:
        current = opened.get()[1]

        print(current.x, current.y)

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
    for eachnode in visited:
        if aNode == eachnode and aNode.f >= eachnode.f:
            return False
    return True


def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (float(z2) - float(z1))**2)


def calculatespeed(t):

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

    else:
        return -1


def findWaterEdges():

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

'''
neighbours = [(xn - 1, yn), (xn + 1, yn), (xn, yn - 1), (xn, yn + 1)]

        for neighbour in neighbours:

            xn, yn = neighbour

            if xn >= row or xn < 0 or yn >= col or yn < 0:
                continue

'''

            #print(neighbour)



def mudaccumulations():
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
                pass



if winter:
    findWaterEdges()
    rgbterrain.save("water.png")

if spring:
    mudaccumulations()



for i in range(len(events)):
    if i < len(events) - 1:
        path = Astar(i, i+1)
        print(path)
        for j in path:
            rgbterrain.putpixel((j[0], j[1]), (255, 0, 0))
        rgbterrain.save("changed.png")


