import numpy as np
from structure.point import Point
from structure.constant import EQN2


class Hyperplane:
    def __init__(self, dim=None, norm=None, offset=None, p1=None, p2=None, is_selected=None):
        if p1 is not None and p2 is not None:
            self.dim = p1.dim
            self.norm = p1.coord - p2.coord
            self.offset = 0
            self.p1 = p1
            self.p2 = p2
            self.is_selected = 0
        if dim is not None and norm is not None and offset is not None:
            self.dim = dim
            self.norm = norm
            self.offset = offset
            self.is_selected = 0
        if dim is not None:
            self.dim = dim
            self.norm = np.zeros(dim, dtype=float)
            self.offset = 0
            self.is_selected = 0

    def print(self):
        print(f"norm = {self.norm}, offset = {self.offset}")

    # check the point is on which side of the hyper-plane
    def check_position(self, p: Point):
        summ = 0
        for i in range(self.dim):
            summ += self.norm[i] * p.coord[i]
        summ += self.offset
        # print(summ)
        if summ >= EQN2:
            return 1
        elif summ <= -EQN2:
            return -1
        else:
            return 0

    # check the distance from the point to the hyper-plane
    def check_distance(self, p: Point):
        numerato = 0
        dinominator = 0
        for i in range(self.dim):
            numerato += p.coord[i] * self.norm[i]
            dinominator += self.norm[i] * self.norm[i]
        numerato += self.offset
        if numerato < 0:
            numerato = -numerato
        dinominator = np.sqrt(dinominator)
        return numerato / dinominator

    # check if the two hyper-planes are the same
    def is_same(self, h: 'Hyperplane'):
        for i in range(self.dim):
            if -EQN2 > (self.norm[i] - h.norm[i]) or (self.norm[i] - h.norm[i]) > EQN2:
                return False
        return True

    # calculate the length of the normal vector
    def norm_length(self):
        return np.linalg.norm(self.norm)
