import os
import time
from typing import List
import numpy as np
from structure.others import isZero


class Point:
    def __init__(self, dim=None, id=None, coord: np.ndarray = None):
        if dim is not None and id is None and coord is None:
            self.id = -1
            self.dim = dim
            self.coord = np.zeros(dim, dtype=float)
            self.mark = False
            self.sampledU = None
        elif dim is not None and id is not None and coord is None:
            self.id = id
            self.dim = dim
            self.coord = np.zeros(dim, dtype=float)
            self.mark = False
            self.sampledU = None
        elif dim is not None and id is None and coord is not None:
            self.id = -1
            self.dim = dim
            self.coord = np.array(coord)
            self.mark = False
            self.sampledU = None
        elif dim is not None and id is not None and coord is not None:
            self.id = id
            self.dim = dim
            self.coord = np.array(coord)
            self.mark = False
            self.sampledU = None

    def __add__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        if self.dim != other.dim:
            raise ValueError("Dimensions of the points do not match.")
        new_coord = self.coord + other.coord
        return Point(dim=self.dim, coord=new_coord)

    def __sub__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        if self.dim != other.dim:
            raise ValueError("Dimensions of the points do not match.")
        new_coord = self.coord - other.coord
        return Point(dim=self.dim, coord=new_coord)

    def normalize(self):
        # length
        length = np.linalg.norm(self.coord)
        if length == 0:
            raise ValueError("Cannot normalize a point with zero length")
        self.coord /= length

    def dot_prod(self, other):
        return np.dot(self.coord, other.coord)

    # print the point
    def print(self):
        print("id: {}".format(self.id))
        print("dim: {}".format(self.dim))
        print("coord: {}".format(self.coord))

    def print_coord(self):
        print("coord: {}".format(self.coord))

    def printAlgResult(self, name, numOfQuestion, startTime, preTime):
        endTime = time.time()
        time_cost = endTime - startTime
        print("-" * 87)
        print(f"|{name:>15} |{numOfQuestion:15d} |{time_cost - preTime:15f} |{'Points':>10} |")
        print(f"|{'-' :>15} |{'-' :>15} |{'-' :>15} |{self.id :>10} |")
        print("-" * 87)
    #    with open("../result.txt", "w") as out_cp:  # "a" represents adding to the end of the file
    #        out_cp.write(f"{numOfQuestion}  {time_cost - preTime:.6f}\n")

    def printToFile(self, name, dataset, epsilon, numofQuestion, startTime, rr):
        endTime = time.time()
        with open("../result/" + name + "_" + dataset + "_" + str(epsilon) + "_" + "result.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{numofQuestion}  {endTime - startTime:.6f} {rr} \n")

    def printToFile2(self, name, dataset, epsilon, numofQuestion, startTime, rr, trainning_size, action_size):
        endTime = time.time()
        with open("../result/" + name + "_" + dataset + "_" + str(epsilon) + "_" + str(trainning_size) + "_" + str(action_size) + "_" + "result.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{numofQuestion}  {endTime - startTime:.6f} {rr} \n")

    def printMiddleResultToFile(self, name, dataset, epsilon, numofQuestion, startTime, rr):
        endTime = time.time()
        with open("../result/" + name + "_" + dataset + "_" + str(epsilon) + "_" + "MiddleResult.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{numofQuestion}  {endTime - startTime:.6f} {rr} \n")

    def dominates(self, other):
        # return True if p1 dominates p2
        return (all(x >= y for x, y in zip(self.coord, other.coord))
                and any(x > y for x, y in zip(self.coord, other.coord)))

    def calc_l1_dist(self, pp):
        diff = 0.0
        for i in range(self.dim):
            diff += abs(self.coord[i] - pp.coord[i])
        return diff

    def is_same(self, p):
        if self.dim != p.dim:
            return False
        for i in range(self.dim):
            if not isZero(self.coord[i] - p.coord[i]):
                return False
        return True

    # Euclidean distance
    def euclidean_distance(self, p):
        return np.linalg.norm(self.coord - p.coord)

'''
p = Point(4)
p.coord[0] = 0.111577; p.coord[1] = 0.224432; p.coord[2] = -0.420030; p.coord[3] = -0.263901
q = Point(4, coord=p.coord)
print(q.coord)
'''
