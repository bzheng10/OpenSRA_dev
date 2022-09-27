"""
"""

import numpy as np
from numba import njit, float64, boolean, int32
from numba.types import Tuple


@njit(
    Tuple((float64[:],float64[:],float64[:],float64[:]))(
        float64[:,:],float64,float64,float64,float64,float64
    ),
    fastmath=True,
    cache=True
)
def get_fault_vertices(
    fault_trace, # (x,y,z) m
    strike, # deg
    dip, # deg
    dip_dir, # deg
    z_top, # m
    z_bot # m
):
    """get (x,y) vertices for a plane given trace and fault angles"""
    # fault geom calcs
    dz = z_bot - z_top # m
    dx = dz/np.tan(np.radians(dip)) # m
    sin_dip_dir = np.sin(np.radians(dip_dir))
    cos_dip_dir = np.cos(np.radians(dip_dir))
    if strike >= 0:
        sign = -1
    else:
        sign = 1

    # get top of rupture vertices
    plane_pt1 = fault_trace[0]
    plane_pt2 = fault_trace[1]

    # get bottom of rupture vertices
    plane_pt3 = plane_pt2.copy()
    plane_pt4 = plane_pt1.copy()
    plane_pt3[0] = plane_pt3[0] + sign*dx*sin_dip_dir
    plane_pt3[1] = plane_pt3[1] + sign*dx*cos_dip_dir
    plane_pt4[0] = plane_pt4[0] + sign*dx*sin_dip_dir
    plane_pt4[1] = plane_pt4[1] + sign*dx*cos_dip_dir

    # return
    return plane_pt1, plane_pt2, plane_pt3, plane_pt4


@njit(
    float64(float64[:]),
    fastmath=True,
    cache=True
)
def get_vect_mag(vect):
    """get vector magnitude"""
    return np.sqrt(vect.dot(vect))


@njit(
    Tuple((boolean, int32, float64[:], float64))(
        float64[:,:],
        float64[:],float64[:],float64[:],float64[:],
    ),
    fastmath=True,
    # cache=True
)
def get_well_fault_intersection_and_angle(
    # traces for well
    well_trace,
    # points 1-4 should be oriented in the same direction: CW or CCW
    plane_pt1,
    plane_pt2,
    plane_pt3,
    plane_pt4
):
    """for a given fault plane and well trace, find intersection and crossing angle"""
    # get unit normal vector to plane
    plane_normal = np.cross(plane_pt4 - plane_pt1, plane_pt2 - plane_pt1)
    plane_unit_normal = plane_normal / get_vect_mag(plane_normal)
    
    # loop through well trace to find segment with intersection
    # initialize default outputs
    intersection_on_well = False
    intersection_default = np.asarray([-999.,-999.,-999.])
    segment_ind_at_intersection_default = -999
    fault_angle = -999
    
    # loop through well trace and check each segment
    for i in range(len(well_trace)-1):
        # make well segment
        segment_pt1 = well_trace[i]
        segment_pt2 = well_trace[i+1]
        segment_vector = segment_pt2 - segment_pt1
        
        # get intersection between inf line and inf plane
        d = (plane_pt1 - segment_pt1).dot(plane_unit_normal / plane_unit_normal.dot(segment_vector))
        intersection = segment_pt1 + (d*segment_vector)
        
        # check if intersection is on segment
        intersection_on_segment = False
        if (intersection-segment_pt1).dot(intersection-segment_pt1) <= (segment_vector).dot(segment_vector):
            intersection_on_segment = True
        
        # check if intersection is on finite plane
        # set up determinant with three of the four vertices and the point of intersection
        pt_a = plane_pt1
        pt_c = plane_pt3
        for j in range(2):
            if j == 0:
                # 1) plane pts 1, 2, 3
                pt_b = plane_pt2
            elif j == 1:
                # 2) plane_pts 1, 4, 3
                pt_b = plane_pt4
            
            # vectors between vertices
            vect_ab = pt_b - pt_a
            vect_bc = pt_c - pt_b
            vect_ca = pt_a - pt_c
            
            # vectors from vertices to intersection
            vect_ap = intersection - pt_a
            vect_bp = intersection - pt_b
            vect_cp = intersection - pt_c

            # get cross-product
            cross_vect_ab_ap = np.cross(vect_ab,vect_ap)
            cross_vect_bc_bp = np.cross(vect_bc,vect_bp)
            cross_vect_ca_cp = np.cross(vect_ca,vect_cp)

            # check sign for last component; if all == same sign, then lies in plane
            intersection_within_plane = False
            if np.abs(np.sign(cross_vect_ab_ap[2]) - np.sign(cross_vect_bc_bp[2])) <= 1 and \
            np.abs(np.sign(cross_vect_bc_bp[2]) - np.sign(cross_vect_ca_cp[2])) <= 1 and \
            np.abs(np.sign(cross_vect_ca_cp[2]) - np.sign(cross_vect_ab_ap[2])) <= 1:
                intersection_within_plane = True
            # break loop if True
            if intersection_within_plane:
                break

        # check if intersection is on segment and within plane (valid)
        if intersection_on_segment and intersection_within_plane:
            intersection_on_well = True
            well_segment_vector_at_intersection = segment_vector
            segment_ind_at_intersection = i
            segment_pt1_at_intersection = segment_pt1
            segment_pt2_at_intersection = segment_pt2
            break

    # use segment vector and plane normal to get internal angle, then fault angle = complimentary of internal angle
    if intersection_on_well:
        internal_angle = np.degrees(np.arccos(
            segment_vector.dot(plane_unit_normal) / \
            get_vect_mag(segment_vector) / \
            get_vect_mag(plane_unit_normal)
        ))
        fault_angle = 90 - internal_angle
    else:
        intersection = intersection_default
        segment_ind_at_intersection = segment_ind_at_intersection_default

    # return
    return intersection_on_well, segment_ind_at_intersection, intersection, fault_angle