import heapq
import time
from typing import List
from structure.point import Point
from structure.point_set import PointSet
from structure.hyperplane import Hyperplane
from scipy.spatial import HalfspaceIntersection, ConvexHull
import matplotlib.pyplot as plt
import swiglpk as glp
from structure import constant
from structure.others import isZero
import numpy as np
import cvxpy as cp
from qpsolvers import solve_qp
from structure.rtree import RTree
from structure.rtree import RTreeNode
from collections import deque
from structure.constant import MAX_VALUE
import random
import bisect
import copy
from scipy.optimize import linprog

SCALE = 10000.0


class HyperplaneSet:
    def __init__(self, dim=None, pr=None):
        if dim is None and pr is None:
            self.hyperplanes = []
            self.ext_pts = []
            self.center = None
            self.out_center = None
            self.out_radius = None
            self.in_center = None
            self.in_radius = None
            self.upper_point = None
            self.lower_point = None
        elif dim is not None and pr is None:
            self.hyperplanes = []
            self.ext_pts = []
            self.center = Point(dim)
            self.out_center = Point(dim)
            self.in_center = Point(dim)
            for i in range(dim):
                self.center.coord[i] = 1.0 / dim
                self.out_center.coord[i] = 1.0 / dim
                self.in_center.coord[i] = 1.0 / dim
            self.out_radius = None
            self.in_radius = None
            self.upper_point = None
            self.lower_point = None
            self.initialize_hyperplanes(dim)
            self.set_ext_pts()
        elif dim is None and pr is not None:
            self.hyperplanes = copy.deepcopy(pr.hyperplanes)
            self.ext_pts = copy.deepcopy(pr.ext_pts)
            self.center = copy.deepcopy(pr.center)
            self.out_center = copy.deepcopy(pr.out_center)
            self.in_center = copy.deepcopy(pr.in_center)
            self.out_radius = pr.out_radius
            self.in_radius = pr.in_radius
            self.upper_point = copy.deepcopy(pr.upper_point)
            self.lower_point = copy.deepcopy(pr.lower_point)

    # add the initial hyper-planes
    def initialize_hyperplanes(self, dim):
        for i in range(dim):
            h = Hyperplane(dim)
            for j in range(dim):
                if i == j:
                    h.norm[j] = -1
            self.hyperplanes.append(h)

    # calculate the average of the extreme points
    def average_point(self):
        dim = self.ext_pts[0].dim
        self.center = Point(dim)
        for i in range(dim):
            for j in range(len(self.ext_pts)):
                self.center.coord[i] += self.ext_pts[j].coord[i]
        for i in range(dim):
            self.center.coord[i] /= len(self.ext_pts)

    # check the relationship between a convex hull and a hyper-plane
    def check_relation(self, h) -> int:
        num_ext = len(self.ext_pts)
        if num_ext < 1:
            print("Warning: The extreme point set of the partition is empty.")
            return -2

        positive, negative = 0, 0
        for i in range(num_ext):
            relation = h.check_position(self.ext_pts[i])
            if relation == -1:
                negative += 1
            elif relation == 1:
                positive += 1
            if positive > 0 and negative > 0:
                return 0

        if negative > 0:
            return -1
        else:
            return 1

    # find a feasible point in the convex hull
    def find_feasible(self):
        M = len(self.hyperplanes)
        D = self.hyperplanes[0].dim

        lp = glp.glp_create_prob()
        glp.glp_set_prob_name(lp, "find_feasible")
        glp.glp_set_obj_dir(lp, glp.GLP_MAX)

        # Adding constraints for each hyperplane
        glp.glp_add_rows(lp, M)
        for i in range(M):
            glp.glp_set_row_name(lp, i + 1, f"q{i + 1}")
            glp.glp_set_row_bnds(lp, i + 1, glp.GLP_UP, 0, 0)

        # Adding variables for dimensions and additional variables for feasibility
        glp.glp_add_cols(lp, D + 2)
        for i in range(D):
            glp.glp_set_col_name(lp, i + 1, f"v{i + 1}")
            glp.glp_set_col_bnds(lp, i + 1, glp.GLP_FR, 0.0, 0.0)  # Free variable

        glp.glp_set_col_name(lp, D + 1, f"v{D + 1}")
        glp.glp_set_col_bnds(lp, D + 1, glp.GLP_LO, 0.0, 0.0)  # Non-negative

        glp.glp_set_col_name(lp, D + 2, f"v{D + 2}")
        glp.glp_set_col_bnds(lp, D + 2, glp.GLP_UP, 0.0, D + 1)

        # Objective function coefficient
        for i in range(D + 1):
            glp.glp_set_obj_coef(lp, i + 1, 0.0)
        glp.glp_set_obj_coef(lp, D + 2, 1.0)

        # Setting matrix coefficients
        ia = glp.intArray((D + 2) * M + 1)
        ja = glp.intArray((D + 2) * M + 1)
        ar = glp.doubleArray((D + 2) * M + 1)
        count = 1
        for i in range(1, M+1):
            for j in range(1, D+3):
                ia[count] = i
                ja[count] = j
                if j <= D:
                    ar[count] = self.hyperplanes[i - 1].norm[j - 1]
                elif j == D + 1:
                    ar[count] = self.hyperplanes[i - 1].offset
                elif j == D + 2:
                    ar[count] = 1
                count += 1

        glp.glp_load_matrix(lp, count - 1, ia, ja, ar)

        parm = glp.glp_smcp()
        glp.glp_init_smcp(parm)
        parm.msg_lev = glp.GLP_MSG_OFF
        # Solving the LP
        glp.glp_simplex(lp, parm)

        feasible_pt = Point(D)
        # Get primary values for additional variables w1 and w2
        w1 = glp.glp_get_col_prim(lp, D + 1)
        w2 = glp.glp_get_col_prim(lp, D + 2)

        # Check feasibility conditions
        if w1 < constant.EQN2 or w2 < constant.EQN2 or isZero(w1) or isZero(w2):
            # print("LP feasible error.")
            glp.glp_delete_prob(lp)
            return None

        # Extracting solution and scaling by w1
        for i in range(D):
            v = glp.glp_get_col_prim(lp, i + 1)
            feasible_pt.coord[i] = v / w1

        glp.glp_delete_prob(lp)
        return feasible_pt

    def LP_check(self, h):
        dim = self.hyperplanes[0].dim
        # another bounder
        new_h = Hyperplane(dim)
        for i in range(dim):
            new_h.norm[i] = 1
        new_h.offset = -SCALE
        self.hyperplanes.append(new_h)
        self.hyperplanes.append(h)
        feasible_pt = self.find_feasible()
        self.hyperplanes.pop()
        self.hyperplanes.pop()
        return feasible_pt

    def set_ext_pts(self):
        dim = self.hyperplanes[0].dim
        # another bounder
        new_h = Hyperplane(dim)
        for i in range(dim):
            new_h.norm[i] = 1
        new_h.offset = -SCALE
        self.hyperplanes.append(new_h)

        interior_point = self.find_feasible()
        if interior_point is None:
            self.hyperplanes.pop()
            print("The intersection is infeasible.\n")
            return False
        #interior_point.print()

        halfspaces = []
        for h in self.hyperplanes:
            new_h = np.zeros(dim + 1)
            for i in range(dim):
                new_h[i] = h.norm[i]
            new_h[dim] = h.offset
            halfspaces.append(new_h)
        # print(halfspaces)
        # Create the HalfspaceIntersection object
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hs_intersection = HalfspaceIntersection(np.array(halfspaces), np.array(interior_point.coord))
        except Exception:
            print("HalfspaceIntersection error.")
            self.hyperplanes.pop()
            return False

        # List boundary hyperplanes
        active_hyperplanes = set()
        for simplex in hs_intersection.dual_facets:
            for facet in simplex:
                active_hyperplanes.add(facet)
        self.hyperplanes = [self.hyperplanes[i] for i in active_hyperplanes if self.hyperplanes[i].offset != -SCALE]

        # Extract the vertices of the intersection
        vertices = hs_intersection.intersections
        self.ext_pts.clear()
        for v in vertices:
            sum_coord = 0
            pp = Point(dim)
            for i in range(dim):
                sum_coord += v[i]
                pp.coord[i] = v[i] / SCALE
            if not isZero(sum_coord):
                self.ext_pts.append(pp)
        # for e in self.ext_pts:
        #    print(e.coord)

        # center
        self.average_point()
        # print(self.out_center.coord)
        return True

    # Find the inner sphere of the convex hull
    # sum (a[i]/||a||) + w <= -off/||a||
    def set_inner_sphere(self):
        dim = self.hyperplanes[0].dim

        # another bounder
        new_h = Hyperplane(dim)
        for i in range(dim):
            new_h.norm[i] = 1.0
        new_h.offset = -SCALE
        self.hyperplanes.append(new_h)

        h_size = len(self.hyperplanes)
        len_norm = []
        for i in range(h_size):
            # self.hyperplanes[i].print()
            len_norm.append(self.hyperplanes[i].norm_length())

        lp = glp.glp_create_prob()
        glp.glp_set_prob_name(lp, "inner_sphere")
        glp.glp_set_obj_dir(lp, glp.GLP_MAX)

        # Adding constraints for each hyperplane <=0
        glp.glp_add_rows(lp, h_size)
        for i in range(h_size - 1):
            glp.glp_set_row_name(lp, i + 1, f"q{i + 1}")
            glp.glp_set_row_bnds(lp, i + 1, glp.GLP_UP, 0, - self.hyperplanes[i].offset / len_norm[i])
        glp.glp_set_row_name(lp, h_size, f"q{h_size}")
        glp.glp_set_row_bnds(lp, h_size, glp.GLP_FX, - self.hyperplanes[h_size - 1].offset, - self.hyperplanes[h_size - 1].offset)

        # Adding variables for dimensions and additional variables for feasibility
        glp.glp_add_cols(lp, dim + 1)
        for i in range(dim + 1):
            glp.glp_set_col_name(lp, i + 1, f"v{i + 1}")
            glp.glp_set_col_bnds(lp, i + 1, glp.GLP_LO, 0.0, 0.0)  # >= 0

        # Objective function coefficient
        for i in range(dim):
            glp.glp_set_obj_coef(lp, i + 1, 0.0)
        glp.glp_set_obj_coef(lp, dim + 1, 1.0)

        # Setting matrix coefficients
        ia = glp.intArray((dim + 1) * h_size + 1)
        ja = glp.intArray((dim + 1) * h_size + 1)
        ar = glp.doubleArray((dim + 1) * h_size + 1)
        count = 1
        for i in range(1, h_size):
            for j in range(1, dim + 2):
                ia[count] = i
                ja[count] = j
                if j <= dim:
                    ar[count] = self.hyperplanes[i - 1].norm[j - 1] / len_norm[i - 1]
                elif j == dim + 1:
                    ar[count] = 1
                count += 1

        for j in range(1, dim + 2):
            ia[count] = h_size
            ja[count] = j
            if j <= dim:
                ar[count] = 1
            elif j == dim + 1:
                ar[count] = 0
            count += 1

        glp.glp_load_matrix(lp, count - 1, ia, ja, ar)


        parm = glp.glp_smcp()
        glp.glp_init_smcp(parm)
        parm.msg_lev = glp.GLP_MSG_OFF
        # Solving the LP
        glp.glp_simplex(lp, parm)

        self.in_center = Point(dim)
        # Extracting solution and scaling by w1
        self.in_radius = glp.glp_get_obj_val(lp) / SCALE
        for i in range(dim):
            v = glp.glp_get_col_prim(lp, i + 1)
            self.in_center.coord[i] = v / SCALE

        glp.glp_delete_prob(lp)

        # print(self.hyperplanes[-1].check_distance(self.in_center))
        self.hyperplanes.pop()

    def find_bounding_point(self, dim_index, max_min):
        # number of dimensions
        dim = self.hyperplanes[0].dim

        # another bounder
        new_h = Hyperplane(dim)
        for i in range(dim):
            new_h.norm[i] = 1.0
        new_h.offset = -SCALE
        self.hyperplanes.append(new_h)

        # number of hyper-planes
        h_size = len(self.hyperplanes)

        lp = glp.glp_create_prob()
        glp.glp_set_prob_name(lp, "optimal dimension")
        if max_min > 0:
            glp.glp_set_obj_dir(lp, glp.GLP_MAX)
        else:
            glp.glp_set_obj_dir(lp, glp.GLP_MIN)

        # Adding constraints for each hyperplane <=0
        glp.glp_add_rows(lp, h_size)
        for i in range(h_size - 1):
            glp.glp_set_row_name(lp, i + 1, f"q{i + 1}")
            glp.glp_set_row_bnds(lp, i + 1, glp.GLP_UP, 0, - self.hyperplanes[i].offset)
        glp.glp_set_row_name(lp, h_size, f"q{h_size}")
        glp.glp_set_row_bnds(lp, h_size, glp.GLP_FX, - self.hyperplanes[h_size - 1].offset, - self.hyperplanes[h_size - 1].offset)

        # Adding variables for dimensions and additional variables for feasibility
        glp.glp_add_cols(lp, dim)
        for i in range(dim):
            glp.glp_set_col_name(lp, i + 1, f"v{i + 1}")
            glp.glp_set_col_bnds(lp, i + 1, glp.GLP_LO, 0.0, 0.0)  # Free variable

        # Objective function coefficient
        for i in range(dim):
            if i == dim_index:
                glp.glp_set_obj_coef(lp, i + 1, 1.0)
            else:
                glp.glp_set_obj_coef(lp, i + 1, 0.0)

        # Setting matrix coefficients
        ia = glp.intArray(dim * h_size + 1)
        ja = glp.intArray(dim * h_size + 1)
        ar = glp.doubleArray(dim * h_size + 1)
        count = 1
        for i in range(1, h_size + 1):
            for j in range(1, dim + 1):
                ia[count] = i
                ja[count] = j
                ar[count] = self.hyperplanes[i - 1].norm[j - 1]
                count += 1

        glp.glp_load_matrix(lp, count - 1, ia, ja, ar)

        parm = glp.glp_smcp()
        glp.glp_init_smcp(parm)
        parm.msg_lev = glp.GLP_MSG_OFF
        # Solving the LP
        glp.glp_simplex(lp, parm)

        # Extracting solution and scaling by w1
        dim_value = glp.glp_get_obj_val(lp) / SCALE

        glp.glp_delete_prob(lp)
        self.hyperplanes.pop()

        return dim_value

    def find_two_farthest_points(self, p: Point):
        farthest_point = None
        second_farthest_point = None
        max_dist = float('-inf')
        second_max_dist = float('-inf')

        for e in self.ext_pts:
            dist = np.linalg.norm(e.coord - p.coord)
            if dist > max_dist:
                second_max_dist = max_dist
                second_farthest_point = farthest_point
                max_dist = dist
                farthest_point = e
            elif dist > second_max_dist:
                second_max_dist = dist
                second_farthest_point = e

        return farthest_point, second_farthest_point, max_dist, second_max_dist

    # Find the sphere that covers the convex hull
    # 1/2x^T P x + q^T x <= 0
    # Gx <= h
    def set_outer_sphere(self):
        self.out_center.coord = copy.deepcopy(self.center.coord)
        while 1:
            max_p, secondmax_p, max_dis, secondmax_dis = self.find_two_farthest_points(self.out_center)
            move_len = (max_dis - secondmax_dis) / 2
            if move_len < 0.001:
                self.out_radius = max_dis
                return
            dim = self.hyperplanes[0].dim
            diff_vector = Point(dim)
            for i in range(dim):
                diff_vector.coord = max_p.coord - self.out_center.coord
            diff_vector_len = np.linalg.norm(diff_vector.coord)
            self.out_center.coord += diff_vector.coord / diff_vector_len * move_len


    def is_interact(self, h_origin, epsilon):
        h = copy.deepcopy(h_origin)
        h.norm = h_origin.p1.coord - (1 - epsilon) * h_origin.p2.coord
        exist = None
        dim = self.hyperplanes[0].dim
        # another bounder
        new_h = Hyperplane(dim)
        for i in range(dim):
            new_h.norm[i] = 1
        new_h.offset = -SCALE
        self.hyperplanes.append(new_h)

        if np.dot(h.norm, self.in_center.coord) + h.offset > 0:
            self.hyperplanes.append(h)
            exist = self.find_feasible()
            self.hyperplanes.pop()
        else:
            ht = copy.deepcopy(h)
            for i in range(ht.dim):
                ht.norm[i] = -ht.norm[i]
            ht.offset = -ht.offset
            self.hyperplanes.append(ht)
            exist = self.find_feasible()
            self.hyperplanes.pop()
        self.hyperplanes.pop()
        if exist is None:
            return False
        return True

    '''
    # find mh candidates hyper-planes
    # the distance from the in_center+out_center to the hyper-plane is the smallest
    def find_candidate_hyperplanes(self, pset: PointSet, mh, pivot_pt):
        candidate_hyperplanes = []
        distance_value = []

        maxdist = np.linalg.norm(self.upper_point.coord - self.lower_point.coord)
        center = Point(pset.points[0].dim)
        center.coord = self.in_center.coord + self.out_center.coord
        sampling_count = 10
        sampled_points = random.sample(pset.points, min(sampling_count, len(pset.points)))
        # find the mh hyper-planes
        for i, p1 in enumerate(sampled_points):
            for j, p2 in enumerate(sampled_points):
                if i <= j:
                    continue
                h = Hyperplane(p1=p1, p2=p2)
                distance = h.check_distance(center)
                if distance < maxdist:
                    pos = bisect.bisect_left(distance_value, distance)
                    if pos < mh:
                        distance_value.insert(pos, distance)
                        candidate_hyperplanes.insert(pos, h)
                        if len(distance_value) > mh:
                            distance_value.pop()
                            candidate_hyperplanes.pop()

        for h in candidate_hyperplanes:
            if self.is_interact(h) is False:
                candidate_hyperplanes.remove(h)
        if len(candidate_hyperplanes) > 0:
            return candidate_hyperplanes

        p1 = pivot_pt
        for j in range(0, len(pset.points)):
            p2 = pset.points[j]
            if p1.id == p2.id:
                continue
            h = Hyperplane(p1=p1, p2=p2)
            if self.is_interact(h):
                distance = h.check_distance(center)
                if distance < maxdist:
                    pos = bisect.bisect_left(distance_value, distance)
                    if pos < mh:
                        distance_value.insert(pos, distance)
                        candidate_hyperplanes.insert(pos, h)
                        if len(distance_value) > mh:
                            distance_value.pop()
                            candidate_hyperplanes.pop()
                            return candidate_hyperplanes

        return candidate_hyperplanes
    '''

    def find_candidate_hyperplanes(self, pset: PointSet, mh, pivot_pt, epsilon):
        candidate_hyperplanes = []
        distance_value = []

        maxdist = np.linalg.norm(self.upper_point.coord - self.lower_point.coord)
        center = Point(pset.points[0].dim)
        center.coord = self.in_center.coord
        sampling_count = 100
        sampled_points = random.sample(pset.points, min(sampling_count, len(pset.points)))
        # sampled_points.append(pivot_pt)
        # find the mh hyper-planes
        for i, p1 in enumerate(sampled_points):
            for j, p2 in enumerate(sampled_points):
                if i <= j or p1.id == p2.id:
                    continue
                h = Hyperplane(p1=p1, p2=p2)
                distance = h.check_distance(center)
                if distance < maxdist:
                    pos = bisect.bisect_left(distance_value, distance)
                    if pos < mh:
                        distance_value.insert(pos, distance)
                        candidate_hyperplanes.insert(pos, h)
                        if len(distance_value) > mh:
                            distance_value.pop()
                            candidate_hyperplanes.pop()

        candidate_hyperplanes = [
            hyperplane for hyperplane in candidate_hyperplanes
            if self.is_interact(hyperplane, 0) is True
        ]
        if len(candidate_hyperplanes) > 0:
            return candidate_hyperplanes

        p1 = pivot_pt
        for j in range(0, len(pset.points)):
            p2 = pset.points[j]
            if p1.id == p2.id:
                continue
            h = Hyperplane(p1=p1, p2=p2)
            if self.is_interact(h, epsilon):
                candidate_hyperplanes.append(h)
                return candidate_hyperplanes

        return candidate_hyperplanes

    def find_candidate_hyperplanes_random(self, pset: PointSet, mh, pivot_pt, epsilon):
        candidate_hyperplanes = []
        distance_value = []

        maxdist = np.linalg.norm(self.upper_point.coord - self.lower_point.coord)
        center = Point(pset.points[0].dim)
        center.coord = self.in_center.coord
        sampling_count = 100
        sampled_points = random.sample(pset.points, min(sampling_count, len(pset.points)))
        # sampled_points.append(pivot_pt)
        # find the mh hyper-planes
        while len(candidate_hyperplanes) < mh:
            # randomly select two points，p1 和 p2
            p1, p2 = random.sample(sampled_points, 2)
            if p1.id == p2.id:
                continue
            h = Hyperplane(p1=p1, p2=p2)
            candidate_hyperplanes.append(h)

        candidate_hyperplanes = [
            hyperplane for hyperplane in candidate_hyperplanes
            if self.is_interact(hyperplane, 0) is True
        ]
        if len(candidate_hyperplanes) > 0:
            return candidate_hyperplanes

        p1 = pivot_pt
        for j in range(0, len(pset.points)):
            p2 = pset.points[j]
            if p1.id == p2.id:
                continue
            h = Hyperplane(p1=p1, p2=p2)
            if self.is_interact(h, epsilon):
                candidate_hyperplanes.append(h)
                return candidate_hyperplanes

        return candidate_hyperplanes


    '''
    def find_candidate_hyperplanes(self, pset: PointSet, mh, pivot_pt, epsilon):
        candidate_hyperplanes = []
        distance_value = []

        maxdist = np.linalg.norm(self.upper_point.coord - self.lower_point.coord)
        center = Point(pset.points[0].dim)
        center.coord = self.in_center.coord + self.out_center.coord
        sampling_count = 100
        sampled_points = random.sample(pset.points, min(sampling_count, len(pset.points)))
        # find the mh hyper-planes
        p1 = pivot_pt
        for j, p2 in enumerate(sampled_points):
            if p1.id == p2.id:
                continue
            h = Hyperplane(p1=p1, p2=p2)
            distance = h.check_distance(center)
            if distance < maxdist:
                pos = bisect.bisect_left(distance_value, distance)
                if pos < mh:
                    distance_value.insert(pos, distance)
                    candidate_hyperplanes.insert(pos, h)
                    if len(distance_value) > mh:
                        distance_value.pop()
                        candidate_hyperplanes.pop()

        candidate_hyperplanes = [
            hyperplane for hyperplane in candidate_hyperplanes
            if self.is_interact(hyperplane, epsilon) is True
        ]
        print(len(candidate_hyperplanes))
        if len(candidate_hyperplanes) > 0:
            return candidate_hyperplanes

        p1 = pivot_pt
        for j in range(0, len(pset.points)):
            p2 = pset.points[j]
            if p1.id == p2.id:
                continue
            h = Hyperplane(p1=p1, p2=p2)
            if self.is_interact(h, epsilon):
                candidate_hyperplanes.append(h)
                return candidate_hyperplanes

        return candidate_hyperplanes
    '''

    def find_candidate_hyperplanes_with_sample_u(self, pset: PointSet, mh):
        candidate_hyperplanes = []
        distance_value = []
        for i in range(mh):
            candidate_hyperplanes.append(None)
            distance_value.append(MAX_VALUE)

        center = Point(pset.points[0].dim)
        center.coord = self.in_center.coord + self.out_center.coord
        while all(value != MAX_VALUE for value in distance_value) is False:
            sampling_count = 1000
            sampled_points = []
            while len(sampled_points) < 10:
                sample_u = self.sample_vector()
                p = pset.find_top_k(sample_u, 1)[0]
                if p not in sampled_points:
                    sampled_points.append(p)
            # find the mh hyper-planes
            for p1 in sampled_points:
                for p2 in sampled_points:
                    if p1.id == p2.id:
                        continue
                    h = Hyperplane(p1=p1, p2=p2)
                    if self.check_relation(h) == 0:
                        # find place to replace
                        distance = h.check_distance(center)
                        pos = bisect.bisect_left(distance_value, distance)
                        if pos < mh:
                            distance_value.insert(pos, distance)
                            candidate_hyperplanes.insert(pos, h)
                            if len(distance_value) > mh:
                                distance_value.pop()
                                candidate_hyperplanes.pop()
        return candidate_hyperplanes

    # find mh candidates hyper-planes
    # the distance from the in_center+out_center to the hyper-plane is the smallest
    def find_candidate_hyperplanes2(self, pset: PointSet, pivot: Point, mh):
        candidate_hyperplanes = []
        distance_value = []
        for i in range(mh):
            candidate_hyperplanes.append(None)
            distance_value.append(MAX_VALUE)

        center = Point(pset.points[0].dim)
        center.coord = self.in_center.coord + self.out_center.coord
        while all(value != MAX_VALUE for value in distance_value) is False:
            sampling_count = 100
            sampled_points = random.sample(pset.points, min(sampling_count, len(pset.points)))
            # find the mh hyper-planes
            for p1 in sampled_points:
                if p1.id == pivot.id:
                    continue
                h = Hyperplane(p1=p1, p2=pivot)
                if self.check_relation(h) == 0:
                    # find place to replace
                    distance = h.check_distance(center)
                    pos = bisect.bisect_left(distance_value, distance)
                    if pos < mh:
                        distance_value.insert(pos, distance)
                        candidate_hyperplanes.insert(pos, h)
                        if len(distance_value) > mh:
                            distance_value.pop()
                            candidate_hyperplanes.pop()
        return candidate_hyperplanes

    # check the stopping condition
    def check_stopping_condition(self, pset: PointSet, epsilon):
        self.set_ext_pts()
        top = pset.find_top_k(self.in_center, 1)
        q = top[0]
        for i in range(q.mark, len(pset.points)):
            p = pset.points[i]
            if q.id != p.id:
                norm = q.coord - (1 - epsilon) * p.coord
                for e in self.ext_pts:
                    numerator = np.dot(norm, e.coord)
                    if np.dot(norm, e.coord) < -0.01:
                        q.mark = i
                        return None
        return q

    # sample a utility vector in the range based on the extreme points
    def sample_vector(self):
        sample_u = Point(self.ext_pts[0].dim)
        weights = []
        sum_w = 0
        for i in range(len(self.ext_pts)):
            weight = np.random.rand()
            weights.append(weight)
            sum_w += weight
        for i in range(len(self.ext_pts)):
            sample_u.coord += self.ext_pts[i].coord * weights[i] / sum_w
        return sample_u

    def select_ext_pts(self, me):
        selected_points = []

        # if the number of extreme points is less than me, return all the extreme points
        if len(self.ext_pts) <= me:
            for e in self.ext_pts:
                selected_points.append(e)
            return selected_points

        # find the radius
        max_dis = 0
        for e in self.ext_pts:
            dis = np.linalg.norm(e.coord - self.center.coord)
            if dis > max_dis:
                max_dis = dis
        circle_radius = max_dis / me

        # find the extreme points that are close to each other
        distance_map = {}
        for i, e1 in enumerate(self.ext_pts):
            for j, e2 in enumerate(self.ext_pts):
                if i < j:
                    dis = np.linalg.norm(e1.coord - e2.coord)
                    if dis < circle_radius:
                        if e1 not in distance_map:
                            distance_map[e1] = []
                        if e2 not in distance_map:
                            distance_map[e2] = []
                        distance_map[e1].append(e2)
                        distance_map[e2].append(e1)
                    else:
                        if e1 not in distance_map:
                            distance_map[e1] = []
                        if e2 not in distance_map:
                            distance_map[e2] = []

        while len(selected_points) < me:
            max_neighbors_point = max(distance_map, key=lambda k: len(distance_map[k]))
            selected_points.append(max_neighbors_point)
            max_neighbors = distance_map.pop(max_neighbors_point)

            for point in distance_map:
                distance_map[point] = [p for p in distance_map[point] if p not in max_neighbors]

        return selected_points

    def random_select_ext_pts(self, me):
        selected_points = []

        # if the number of extreme points is less than me, return all the extreme points
        if len(self.ext_pts) <= me:
            for e in self.ext_pts:
                selected_points.append(e)
            return selected_points

        selected_points = random.sample(self.ext_pts, me)

        return selected_points

    def select_poly(self, poly_set, mp):
        selected_poly = []

        # if the number of extreme points is less than me, return all the extreme points
        if len(poly_set) <= mp:
            for poly in poly_set:
                selected_poly.append(poly)
            return selected_poly

        # find the extreme points that are close to each other
        connect_map = {}
        for i, p1 in enumerate(poly_set):
            for j, p2 in enumerate(poly_set):
                if i < j:
                    dis = np.linalg.norm(p1.out_center.coord - p2.out_center.coord)
                    if dis < p1.out_radius + p2.out_radius:
                        if p1 not in connect_map:
                            connect_map[p1] = []
                        if p2 not in connect_map:
                            connect_map[p2] = []
                        connect_map[p1].append(p2)
                        connect_map[p2].append(p1)
                    else:
                        if p1 not in connect_map:
                            connect_map[p1] = []
                        if p2 not in connect_map:
                            connect_map[p2] = []

        while len(selected_poly) < mp:
            max_neighbors_poly = max(connect_map, key=lambda k: len(connect_map[k]))
            selected_poly.append(max_neighbors_poly)
            max_neighbors = connect_map.pop(max_neighbors_poly)

            for poly in connect_map:
                connect_map[poly] = [p for p in connect_map[poly] if p not in max_neighbors]

        return selected_poly

    def is_inside(self, p: Point):
        for h in self.hyperplanes:
            if np.dot(h.norm, p.coord) + h.offset > constant.EQN2:
                return False
        return True

    def get_state_high(self):
        dim = self.hyperplanes[0].dim
        self.set_inner_sphere()
        # bounding box
        self.upper_point = Point(dim)
        self.lower_point = Point(dim)
        for i in range(dim):
            self.upper_point.coord[i] = self.find_bounding_point(i, 1)
            self.lower_point.coord[i] = self.find_bounding_point(i, 0)
        state = np.concatenate((self.in_center.coord, np.array([self.in_radius]), self.upper_point.coord, self.lower_point.coord)) # 3dim+1
        return state

    def get_state_low(self, pset: PointSet, dim, epsilon, me, mp):
        poly_set = {}
        p_candidate = []
        for i in range(dim * 10):
            sample_u = self.sample_vector()
            # check if the sample is inside any polyhedron
            is_exist = False
            for py in poly_set:
                if py.is_inside(sample_u):
                    poly_set[py] += 1
                    is_exist = True
            if is_exist:
                continue
            # if not, build a new polyhedron
            p = pset.find_top_k(sample_u, 1)[0]
            p_candidate.append(p)
            poly = HyperplaneSet(pr=self)
            for q in pset.points:
                if q.is_same(p):
                    continue
                h = Hyperplane(dim)
                h.norm = (1 - epsilon) * q.coord - p.coord
                poly.hyperplanes.append(h)
            poly.set_ext_pts()
            poly_set[poly] = 1

        for sample_u in self.ext_pts:
            # check if the sample is inside any polyhedron
            is_exist = False
            for py in poly_set:
                if py.is_inside(sample_u):
                    poly_set[py] += 1
                    is_exist = True
            if is_exist:
                continue
            # if not, build a new polyhedron
            p = pset.find_top_k(sample_u, 1)[0]
            p_candidate.append(p)
            poly = HyperplaneSet(pr=self)
            for q in pset.points:
                if q.is_same(p):
                    continue
                h = Hyperplane(dim)
                h.norm = (1 - epsilon) * q.coord - p.coord
                poly.hyperplanes.append(h)
            poly.set_ext_pts()
            poly_set[poly] = 1

        # find the top mp points
        top_m_polys = []
        if len(poly_set) < mp:
            top_m_polys = heapq.nlargest(len(poly_set), poly_set.items(), key=lambda item: item[1])
        else:
            top_m_polys = heapq.nlargest(mp, poly_set.items(), key=lambda item: item[1])
        state = []
        for poly, count in top_m_polys:
            poly.set_outer_sphere()
            state.extend(poly.out_center.coord)
            state.append(poly.out_radius)
            for rep_e in poly.select_ext_pts(me=me):
                state.extend(rep_e.coord)

        expected_length = mp * (dim + 1 + dim * me)
        if len(state) < expected_length:
            state.extend([0] * (expected_length - len(state)))

        state = np.array(state)
        return p_candidate, state

    def get_state_low_abla(self, pset: PointSet, dim, epsilon, me, mp, action_space_size):
        poly_set = {}
        p_candidate = []
        for i in range(dim * 10):
            sample_u = self.sample_vector()
            # check if the sample is inside any polyhedron
            is_exist = False
            for py in poly_set:
                if py.is_inside(sample_u):
                    poly_set[py] += 1
                    is_exist = True
            if is_exist:
                continue
            # if not, build a new polyhedron
            p = pset.find_top_k(sample_u, 1)[0]
            p_candidate.append(p)
            poly = HyperplaneSet(pr=self)
            for q in pset.points:
                if q.is_same(p):
                    continue
                h = Hyperplane(dim)
                h.norm = (1 - epsilon) * q.coord - p.coord
                poly.hyperplanes.append(h)
            poly.set_ext_pts()
            poly_set[poly] = 1

        for sample_u in self.ext_pts:
            # check if the sample is inside any polyhedron
            is_exist = False
            for py in poly_set:
                if py.is_inside(sample_u):
                    poly_set[py] += 1
                    is_exist = True
            if is_exist:
                continue
            # if not, build a new polyhedron
            p = pset.find_top_k(sample_u, 1)[0]
            p_candidate.append(p)
            poly = HyperplaneSet(pr=self)
            for q in pset.points:
                if q.is_same(p):
                    continue
                h = Hyperplane(dim)
                h.norm = (1 - epsilon) * q.coord - p.coord
                poly.hyperplanes.append(h)
            poly.set_ext_pts()
            poly_set[poly] = 1

        state = []
        self.set_outer_sphere()
        state.extend(self.out_center.coord)
        state.append(self.out_radius)
        for rep_e in self.select_ext_pts(me=me):
            state.extend(rep_e.coord)

        expected_length = dim + 1 + dim * me
        if len(state) < expected_length:
            state.extend([0] * (expected_length - len(state)))

        state = np.array(state)
        return p_candidate, state


    def get_state_low2(self, pset: PointSet, dim, epsilon, me, mp):
        poly_set = {}
        p_candidate = []
        for i in range(dim * 10):
            sample_u = self.sample_vector()
            # check if the sample is inside any polyhedron
            is_exist = False
            for py in poly_set:
                if py.is_inside(sample_u):
                    poly_set[py] += 1
                    is_exist = True
            if is_exist:
                continue
            # if not, build a new polyhedron
            p = pset.find_top_k(sample_u, 1)[0]
            p_candidate.append(p)
            poly = HyperplaneSet(pr=self)
            for q in pset.points:
                if q.is_same(p):
                    continue
                h = Hyperplane(dim)
                h.norm = (1 - epsilon) * q.coord - p.coord
                poly.hyperplanes.append(h)
            poly.set_ext_pts()
            poly.set_outer_sphere()
            poly_set[poly] = 1

        for sample_u in self.ext_pts:
            # check if the sample is inside any polyhedron
            is_exist = False
            for py in poly_set:
                if py.is_inside(sample_u):
                    poly_set[py] += 1
                    is_exist = True
            if is_exist:
                continue
            # if not, build a new polyhedron
            p = pset.find_top_k(sample_u, 1)[0]
            p_candidate.append(p)
            poly = HyperplaneSet(pr=self)
            for q in pset.points:
                if q.is_same(p):
                    continue
                h = Hyperplane(dim)
                h.norm = (1 - epsilon) * q.coord - p.coord
                poly.hyperplanes.append(h)
            poly.set_ext_pts()
            poly.set_outer_sphere()
            poly_set[poly] = 1

        # find the top mp points
        top_m_polys = self.select_poly(poly_set, mp)
        state = []
        for poly in top_m_polys:
            state.extend(poly.out_center.coord)
            state.append(poly.out_radius)
            for rep_e in poly.random_select_ext_pts(me=me):
                state.extend(rep_e.coord)

        expected_length = mp * (dim + 1 + dim * me)
        if len(state) < expected_length:
            state.extend([0] * (expected_length - len(state)))

        state = np.array(state)
        return p_candidate, state

    def get_rrbound_approx(self):
        if len(self.ext_pts) <= 0:
            return 1
        dim = self.ext_pts[0].dim

        # compute the Minimum Bounding Rectangle(MBR)
        maxd = np.zeros(dim)
        mind = np.zeros(dim)
        for i in range(dim):
            maxd[i] = self.ext_pts[0].coord[i]
            mind[i] = self.ext_pts[0].coord[i]

        for e in self.ext_pts:
            for j in range(dim):
                if e.coord[j] > maxd[j]:
                    maxd[j] = e.coord[j]
                elif e.coord[j] < mind[j]:
                    mind[j] = e.coord[j]

        # bound calculation
        bound = 0.0
        for i in range(dim):
            bound += maxd[i] - mind[i]
        bound *= dim

        if bound < 1.0:
            return bound
        else:
            return 1.0

    def get_rrbound_exact(self):
        if len(self.ext_pts) <= 0:
            return 1

        maxL = 0.0
        # find the maximum pairwise L1-distance between the extreme vertices of R
        for i in range(len(self.ext_pts)):
            for j in range(i + 1, len(self.ext_pts)):
                v = self.ext_pts[i].calc_l1_dist(self.ext_pts[j])
                if v > maxL:
                    maxL = v

        maxL *= self.ext_pts[0].dim
        if maxL < 1.0:
            return maxL
        else:
            return 1.0

    def dom(self, p1: Point, p2: Point):
        normal = p1.__sub__(p2)
        below_count = 0
        for e in self.ext_pts:
            value = normal.dot_prod(e)
            if value < 0 and not isZero(value):
                below_count += 1
                break

        if below_count <= 0:
            return 1
        else:
            return 0

    def rtree_prune(self, pset: List[Point], C_idx: List[int], stop_option):
        rr = 1.0
        if stop_option == constant.EXACT_BOUND:
            rr = self.get_rrbound_exact()
        elif stop_option == constant.APPROX_BOUND:
            rr = self.get_rrbound_approx()

        # build r-tree
        dim = pset[0].dim
        rt = RTree(dim)
        if len(C_idx) <= 1:
            return C_idx, rr
        for idx in C_idx:
            rt.insert(pset[idx])
        node_queue = deque()
        node_queue.append(rt.root)
        sl = []
        while len(node_queue) > 0:
            n = node_queue.popleft()
            if not n.is_leaf:
                dominated = False
                node_pt = Point(dim, [n.max_point[i] for i in range(dim)])
                for j in range(len(sl)):
                    if self.dom(pset[sl[j]], node_pt):
                        dominated = True
                        break

                if not dominated:
                    for child in n.entries:
                        node_queue.append(child)
            else:
                for child in n.entries:
                    idx = child.id
                    dominated = False
                    for j in range(len(sl)):
                        if self.dom(pset[sl[j]], pset[idx]):
                            dominated = True
                            break

                    if not dominated:
                        sl = [j for j in sl if not self.dom(pset[idx], pset[j])]
                        sl.append(idx)

        # Clean up
        C_idx.clear()
        C_idx.extend(sl)

        return C_idx, rr

    def print_hs(self):
        print("Utility Range: ")
        for h in self.hyperplanes:
            h.print()
        for e in self.ext_pts:
            e.print_coord()
        #print("in sphere: ", self.in_center.coord, self.in_radius)
        #print("out sphere: ", self.out_center.coord, self.out_radius)
        #print(" ")

    def cal_regret(self, pset: PointSet, alg_name, dataset_name, num_question):
        self.set_ext_pts()
        self.set_inner_sphere()
        avg_u = self.in_center
        p_return = pset.find_top_k(avg_u, 1)[0]

        max_regret = -1
        for i in range(1, 10000):
            sample_u = self.sample_vector()
            p_top = pset.find_top_k(sample_u, 1)[0]
            v1 = sample_u.dot_prod(p_top)
            v2 = sample_u.dot_prod(p_return)
            regret = (v1 - v2) / v1
            if regret > max_regret:
                max_regret = regret

        for i in range(len(self.ext_pts)):
            sample_u = self.ext_pts[i]
            p_top = pset.find_top_k(sample_u, 1)[0]
            v1 = sample_u.dot_prod(p_top)
            v2 = sample_u.dot_prod(p_return)
            regret = (v1 - v2) / v1
            if regret > max_regret:
                max_regret = regret

        print(f"{num_question}  {max_regret:.6f} \n")
        with open("../result/" + "_" + alg_name + "_" + dataset_name + "_" + "regret_result.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{num_question}  {max_regret:.6f} \n")
        out_cp.close()

        return max_regret

    def cal_regret_hit_and_run_sample(self, pset: PointSet, alg_name, dataset_name, num_question):
        dim = self.hyperplanes[0].dim
        self.set_inner_sphere()
        avg_u = self.in_center
        p_return = pset.find_top_k(avg_u, 1)[0]
        max_regret = -1
        num_samples = 10000
        for nn in range(num_samples):
            if nn % 500 == 0:
                print(nn)
            sample_u = Point(dim)
            sample_u.coord = np.abs(np.random.randn(dim))
            sample_u.coord /= sample_u.coord.sum()
            while self.is_inside(sample_u) is False:
                sample_u.coord += (self.in_center.coord - sample_u.coord) / np.random.uniform(1, 2)

            p_top = pset.find_top_k(sample_u, 1)[0]
            v1 = sample_u.dot_prod(p_top)
            v2 = sample_u.dot_prod(p_return)
            regret = (v1 - v2) / v1
            if regret > max_regret:
                max_regret = regret

        print(f"{num_question}  {max_regret:.6f} \n")
        with open("../result/" + "_" + alg_name + "_" + dataset_name + "_" + "regret_result.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{num_question}  {max_regret:.6f} \n")
        out_cp.close()

        return max_regret

    def print_time(self, alg_name, dataset_name, numofQuestion, startTime):
        endTime = time.time()
        with open("../result/" + "_" + alg_name + "_" + dataset_name + "_" + "time_result.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{numofQuestion}  {endTime - startTime:.6f} \n")
        out_cp.close()
        print(f"{numofQuestion}  {endTime - startTime:.6f} \n")

    def print_point_traj(self, dataset_name, index, numofQuestion, p: Point):
        with open(f"../{dataset_name}/{index}.txt", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"{p.coord} {numofQuestion} \n")
        out_cp.close()
        print(f"{p.coord} {numofQuestion} \n")

