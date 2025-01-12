# pylint: disable=trailing-whitespace,bad-whitespace,invalid-name

# File: vecMath.py
# Auth: G. E. Deschaines
# Date: 18 May 2015
# Prog: Assortment of vector and matrix math functions.
# Desc: Based on vector math functions developed by Matt Heinzen and documented 
#       in his "PyODE Ragdoll Physics Tutorial" which can be viewed and obtained 
#       from the following links.
#
#         http://www.monsterden.net/software/ragdoll-pyode-tutorial
#         http://www.monsterden.net/software/download/ragdoll-pyode-tutorial.py
#
# Note: The vector and matrix math functions herein are ONLY applicable to 
#       1x3 vectors and 3x3 matrices.
#
# Disclaimer:
#
# See DISCLAIMER

from math import pi, sqrt, acos, atan2, sin, cos

# Math Conversion Factors

RPD    = pi/180.0
DPR    = 180.0/pi
TWOPI  = 2.0*pi
HALFPI = pi/2.0
INF    = float('inf')

# Vector Math Functions

def vecMulS(V,s):
    """
    Returns vector result of multiplying vector V by scalar s.
    """
    return (V[0]*s, V[1]*s, V[2]*s)
  
def vecDivS(V,s):
    """
    Returns vector result of dividing vector V by scalar s.
    """
    if s == 0.0 : return (INF, INF, INF)
    return (float(V[0])/s, float(V[1])/s, float(V[2])/s)
  
def vecAdd(A,B):
    """
    Returns vector result of adding vectors A and B.
    """
    return (A[0]+B[0], A[1]+B[1], A[2]+B[2])
  
def vecSub(A,B):
    """
    Returns vector result of subtracting vector B from vector A.
    """
    return (A[0]-B[0], A[1]-B[1], A[2]-B[2])
    
def vecMagSq(V):
    """
    Returns scalar magnitude squared of vector V.
    """
    try:
        vsq = V[0]**2 + V[1]**2 + V[2]**2
    except OverflowError:
        vsq = 1.0
    return (vsq)
  
def vecMag(V):
    """
    Returns scalar magnitude of vector V.
    """
    magsq = vecMagSq(V)
    if magsq > 0.0 : return sqrt(magsq)
    return 0.0

def unitVec(V):
    """
    Returns unit vector for vector V.
    """
    mag = vecMag(V)
    if mag > 0.0 : return vecMulS(V,1.0/mag)
    return (0.0, 0.0, 0.0)
    
def vecDotP(A,B):
    """
    Returns scalar dot product of vector A with vector B.
    """
    return (A[0]*B[0] + A[1]*B[1] + A[2]*B[2])
    
def acosUVecDotP(UA,UB):
    """
    Returns scalar arccosine (rad) of unit vectors UA and UB.
    """
    cosang = vecDotP(UA,UB)
    if cosang < -1.0 : return pi
    if cosang >  1.0 : return 0.0
    return acos(cosang)
    
def acosVecDotP(A,B):
    """
    Returns scalar arccosine (rad) of vectors A and B.
    """
    UA = unitVec(A)
    UB = unitVec(B)
    return acosUVecDotP(UA,UB)
  
def projectionVecAonUB(A,UB):
    """
    Returns projection vector of vector A on unit vector B.
    """
    return vecMulS(UB,vecDotP(A,UB))
    
def rejectionVecAfromUB(A,UB):
    """
    Returns rejection vector of vector A from unit vector B.
    """
    return vecSub(A,projectionVecAonUB(A,UB))
  
def vecCrossP(A,B):
    """
    Returns vector cross product of vector A with vector B.
    """
    return ( A[1]*B[2] - B[1]*A[2],\
              A[2]*B[0] - B[2]*A[0],\
              A[0]*B[1] - B[0]*A[1] )
         
def atanVecCrossP(A,B):
    """
    Returns scalar arctangent (rad) of cross product of vectors A and B.
    """
    x = vecDotP(A,B)
    y = vecMag(vecCrossP(A,B))
    return atan2(y, x)
  
def unitNormalVecFromAtoB(A,B):
    """
    Returns unit normal vector from vector A to vector B.
    """
    return unitVec(vecCrossP(A,B))
  
def unitNormalVecAndAngleFromAtoB(A,B):
    """
    Returns unit normal vector and angle (rad) from vector A to vector B.
    """
    Nvec = vecCrossP(A,B)
    return ( unitVec(Nvec), atan2(vecMag(Nvec),vecDotP(A,B)) )
    
def vecRotZ(V, rot):
    """
    Performs 2D transformation on given XYZ vector 'V' to yield an 
    xyz vector in a coordinate frame rotated about the Z/z axis by
    the given 'rot' angle in radians.
    """
    cosr = cos(rot)
    sinr = sin(rot)
      
    x =  cosr*V[0] + sinr*V[1] 
    y = -sinr*V[0] + cosr*V[1]
    z = V[2]
      
    return (x, y, z)
 
def vecToDiagMat(V):
    """
    Returns square diagonal matrix with vector V along diagonal.
    """
    row0 = (V[0],  0.0,  0.0)
    row1 = ( 0.0, V[1],  0.0)
    row2 = ( 0.0,  0.0, V[2])
    return (row0, row1, row2)
    
def vecToSkewMat(V):
    """
    Returns square skew symmetric matrix from vector V.
    """
    row0 = (  0.0, -V[2],  V[1])
    row1 = ( V[2],   0.0, -V[0])
    row2 = (-V[1],  V[0],   0.0)
    return (row0, row1, row2)

def vecMulV(A,B):
    """
    Returns matrix from elememt-wise multiplication of vectors A and B.
    """
    row0 = (A[0]*B[0], A[0]*B[1], A[0]*B[2])
    row1 = (A[1]*B[0], A[1]*B[1], A[1]*B[2])
    row2 = (A[2]*B[0], A[2]*B[1], A[2]*B[2])
    return (row0, row1, row2)
    
def vecMulM(V,M):
    """
    Returns matrix from multiplication of vector V with matrix M.
    """
    v0 = (V[0]*M[0][0] + V[1]*M[1][0] + V[2]*M[2][0])
    v1 = (V[0]*M[0][1] + V[1]*M[1][1] + V[2]*M[2][1])
    v2 = (V[0]*M[0][2] + V[1]*M[1][2] + V[2]*M[2][2])
    return (v0, v1, v2)
    
def matAdd(A,B):
    """
    Returns matrix from elememt-wise addition of matrices A and B.
    """
    row0 = (A[0][0]+B[0][0], A[0][1]+B[0][1], A[0][2]+B[0][2])
    row1 = (A[1][0]+B[1][0], A[1][1]+B[1][1], A[1][2]+B[1][2])
    row2 = (A[2][0]+B[2][0], A[2][1]+B[2][1], A[2][2]+B[2][2])
    return (row0, row1, row2)
    
def matSub(A,B):
    """
    Returns matrix from elememt-wise subtraction of matrix B from matrix A.
    """
    row0 = (A[0][0]-B[0][0], A[0][1]-B[0][1], A[0][2]-B[0][2])
    row1 = (A[1][0]-B[1][0], A[1][1]-B[1][1], A[1][2]-B[1][2])
    row2 = (A[2][0]-B[2][0], A[2][1]-B[2][1], A[2][2]-B[2][2])
    return (row0, row1, row2)
    
def matMul(A,B):
    """
    Returns matrix from elememt-wise multiplication of matrices A and B.
    """
    row0 = (A[0][0]*B[0][0], A[0][1]*B[0][1], A[0][2]*B[0][2])
    row1 = (A[1][0]*B[1][0], A[1][1]*B[1][1], A[1][2]*B[1][2])
    row2 = (A[2][0]*B[2][0], A[2][1]*B[2][1], A[2][2]*B[2][2])
    return (row0, row1, row2)

def matSq(M):
    """
    Returns matrix from elememt-wise squaring of matrix M.
    """
    row0 = (M[0][0]**2, M[0][1]**2, M[0][2]**2)
    row1 = (M[1][0]**2, M[1][1]**2, M[1][2]**2)
    row2 = (M[2][0]**2, M[2][1]**2, M[2][2]**2)
    return (row0, row1, row2)
    
def matMulS(M,s):
    """
    Returns matrix from multiplying matrix M by scalar s.
    """
    row0 = (M[0][0]*s, M[0][1]*s, M[0][2]*s)
    row1 = (M[1][0]*s, M[1][1]*s, M[1][2]*s)
    row2 = (M[2][0]*s, M[2][1]*s, M[2][2]*s)
    return (row0, row1, row2)
    
def matDivS(M,s):
    """
    Returns matrix from dividing matrix M by scalar s.
    """
    if s == 0.0 :
        row0 = (INF, INF, INF)
        row1 = (INF, INF, INF)
        row2 = (INF, INF, INF)
    else :
        row0 = (M[0][0]/s, M[0][1]/s, M[0][2]/s)
        row1 = (M[1][0]/s, M[1][1]/s, M[1][2]/s)
        row2 = (M[2][0]/s, M[2][1]/s, M[2][2]/s)
    return (row0, row1, row2)
    
def matDotV(M,V):
    """
    Returns vector from dot product of matrix M with vector V.
    """
    v0 = vecDotP(M[0],V)
    v1 = vecDotP(M[1],V)
    v2 = vecDotP(M[2],V)
    return ( v0, v1, v2 )

def matDotM(A,B):
    """
    Returns matrix from dot product of matrix A with matrix B.
    """
    row0 = (A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
            A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
            A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2])
    row1 = (A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
            A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
            A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2])
    row2 = (A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
            A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
            A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2])
    return (row0, row1, row2)
    
def matSqM(M):
    """
    Returns matrix from dot product of matrix M with itself.
    """
    row0 = (M[0][0]*M[0][0] + M[0][1]*M[1][0] + M[0][2]*M[2][0],
            M[0][0]*M[0][1] + M[0][1]*M[1][1] + M[0][2]*M[2][1],
            M[0][0]*M[0][2] + M[0][1]*M[1][2] + M[0][2]*M[2][2])
    row1 = (M[1][0]*M[0][0] + M[1][1]*M[1][0] + M[1][2]*M[2][0],
            M[1][0]*M[0][1] + M[1][1]*M[1][1] + M[1][2]*M[2][1],
            M[1][0]*M[0][2] + M[1][1]*M[1][2] + M[1][2]*M[2][2])
    row2 = (M[2][0]*M[0][0] + M[2][1]*M[1][0] + M[2][2]*M[2][0],
            M[2][0]*M[0][1] + M[2][1]*M[1][1] + M[2][2]*M[2][1],
            M[2][0]*M[0][2] + M[2][1]*M[1][2] + M[2][2]*M[2][2])
    return (row0, row1, row2)    
    
def transposeM(M):
    """
    Returns matrix transpose of matrix M.
    """
    row0 = (M[0][0], M[1][0], M[2][0])
    row1 = (M[0][1], M[1][1], M[2][1])
    row2 = (M[0][2], M[1][2], M[2][2])
    return (row0, row1, row2)

def xformMatRotZ(angle):
    """
    Returns Cartesian body xyz to world XYZ coordinate frame transformation
    matrix for given angle rotation in radians about Z-axis.
    """
    row0 = (cos(angle), -sin(angle),  0.0)
    row1 = (sin(angle),  cos(angle),  0.0)
    row2 = (       0.0,         0.0,  1.0)
    return (row0, row1, row2)
