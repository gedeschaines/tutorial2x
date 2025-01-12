# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:21:26 2020

@author: garyd
"""
# File: tutorial2util.py
# Auth: Gary E. Deschaines
# Date: 17 Jul 2020
# Prog: Utility procedures common to tutorial2x programs. 
# Desc: Procedures to print and manipulate vectors and matrices associated 
#       with ODE joints and bodies,
#
# Disclaimer:
#
# See DISCLAIMER

from numpy import zeros
from vecMath import vecMag


def printVec(body,name,V):
    """
    Prints elements and magnitude of the named body vector.
    """
    fmt = "%s:  %s = %10.4f (%9.4f | %9.4f | %9.4f)"
    
    print(fmt % (body, name, vecMag(V), V[0], V[1], V[2]) )
    
  
def printFeedback(name, t, fb):
    """
    Prints ODE joint name, simulation time, and ODE joint's feedback force
    and torque values.
    """ 
    F1 = vecMag(fb[0])
    T1 = vecMag(fb[1])
    F2 = vecMag(fb[2])
    T2 = vecMag(fb[3])
    
    print("fb:  %s, t=%8.3f" % (name, t))
    print("  F1:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
            (F1, fb[0][0], fb[0][1], fb[0][2]) )
    print("  T1:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
            (T1, fb[1][0], fb[1][1], fb[1][2]) )
    print("  F2:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
            (F2, fb[2][0], fb[2][1], fb[2][2]) )
    print("  T2:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
            (T2, fb[3][0], fb[3][1], fb[3][2]) )
    
    
def rotToM(R):
    """
    Converts given ODE rotation 9-tuple matrix to row vector matrix
    form used by the vecMath matrix functions.
    """
    mat0 = ( R[0], R[1], R[2] )
    mat1 = ( R[3], R[4], R[5] )
    mat2 = ( R[6], R[7], R[8] )
    
    return (mat0, mat1, mat2)


def bodyRotMatrix(body):
    """
    Returns body to world frame rotation matrix for given ODE body.
    """
    rot = body.getRotation()
    mat = zeros((3,3))
    mat[0,:] = rot[0:3]
    mat[1,:] = rot[3:6]
    mat[2,:] = rot[6:9]
    return mat

