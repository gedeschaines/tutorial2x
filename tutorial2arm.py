#!/usr/bin/env ipython --matplotlib=qt

# pylint: disable=trailing-whitespace,bad-whitespace,invalid-name
# pylint: disable=anomalous-backslash-in-string,bad-continuation
# pylint: disable=multiple-statements,redefined-outer-name,global-statement

# File: tutorial2arm.py
# Auth: Gary E. Deschaines
# Date: 5 July 2015
# Prog: Tutorial on 2R robotic arm dynamics with PyODE and Pygame
# Desc: Models numerical solution for robot dynamics example presented
#       in section 11.6 of reference [1] listed below.
#
# The purpose of this program is to simulate the motion of a planar 
# 2R robotic manipulator as articulated body solids modeled with Open
# Dynamics Engine (ODE) using torques calculated by iteratively solving
# a state space representation of the dynamic equations of motion for
# the modeled system as derived by Newton-Euler recursion.
#
# Original basis for this program was obtained at
#
# https://bitbucket.org/odedevs/ode/src/master/bindings/python/demos/tutorial2.py
#
# and modifications include these changes: double pendulum system to
# planar 2R robotic arm, joints to hinge and bodies to rods, utilized
# NumPy and SciPy linear algebra functions to analytically solve the
# differential equations of motions, added integration of differential
# equations of motion using Runge-Kutta 4th order method (RK4), and 
# utilized Matplotlib to create plots of data collected from the ODE
# simulation and RK4 integration.
#
# References:
#
# [1] Dr. Robert L. William II, "Robot Mechanics: Notesbook Supplement
#     for ME 4290/5290 Mechanics and Control of Robotic Manipulators",
#     Ohio University, Mech. Engineering, Spring 2015; web available at
#     https://people.ohio.edu/williams/html/PDF/Supplement4290.pdf
#
# [2] John J. Craig, "Introduction to Robotics: Mechanics and Control",
#     3rd ed., Pearson Prentice Hall, Pearson Education, Inc., Upper 
#     Saddle River, NJ, 2005
# 
# [3] Kevin M. Lynch and Frank C. Park, "Modern Robotics: Mechanics, Planning
#     and Control", preprint ver., Cambridge University Press, May 3, 2017;
#     web available at http://hades.mech.northwestern.edu/images/7/7f/MR.pdf
#
# Disclaimer:
#
# See DISCLAIMER

import sys

from math import ceil, floor, cos, sin
from locale import format_string

try:
    import ode
except ImportError:
    print("* Error: PyODE package required.")
    sys.exit()
  
try:
    import pygame
    from pygame.locals import QUIT, KEYDOWN
except ImportError:
    print("* Error: PyGame package required.")
    sys.exit()
  
try:
    from vecMath import DPR, RPD
    from vecMath import vecAdd, vecSub, vecCrossP, vecDivS, vecMulS
    from vecMath import vecMag, vecRotZ
    from vecMath import matDotV, vecToSkewMat, transposeM
except ImportError:
    print("* Error: vecMath package required.")
    sys.exit()
  
try:
    import numpy             as np
    import scipy.linalg      as la
    import matplotlib        as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("* Error: NumPy, SciPy and Matplotlib packages required.")
    print("         Suggest installing the SciPy stack.")
    sys.exit()
  
try:
    from RK4_Solver import RK4_Solver
except ImportError:
    print("* Error: RK4_Solver class required.")
    sys.exit()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Processing and output control flags

T_STEP     = 0.001  # Simulation and integration time step size (sec)
T_STOP     = 1.000  # Simulation and integration stop time (sec)
MOTOR_AVEL = False  # Use motor.setParam(ode.ParamVel,...) versus motor.addTorques(...)
PRINT_FBCK = False  # Controls printing of joint feedback data
PRINT_EVAL = False  # Controls printing of dynamics evaluation calculations
PRINT_DATA = False  # Controls printing of collected data
PLOT_DATA  = True   # Controls plotting of collected data
SAVE_ANIM  = False  # Controls saving animation images

# Drawing constants

WINDOW_RESOLUTION = (640, 480)

DRAW_SCALE = WINDOW_RESOLUTION[0] / 4
"""Factor to multiply physical coordinates by to obtain screen size in pixels"""
DRAW_OFFSET = (WINDOW_RESOLUTION[0] / 2, 300) 
"""Screen coordinates (in pixels) that map to the physical origin (0, 0, 0)"""

FILL_COLOR = (255, 255, 255)  # background fill color
TEXT_COLOR = ( 10,  10,  10)  # text drawing color
PATH_WIDTH = 1  # width of end-effector path line
PATH_COLOR = (  0,   0,   0)  # end-effector planned path color
BOX_WIDTH  = 8  # width of the line (in pixels) representing a box
BOX_COLOR  = (  0,  50, 200)  # for drawing box orientations from ODE
BDY_COLOR  = (  0,  50, 200)  # for drawing body positions from ODE
JNT_COLOR  = ( 50,  50,  50)  # for drawing joint positions from ODE
ROD_WIDTH  = 4  # width of the line (in pixels) representing a rod
ROD_COLOR  = (255,   0, 255)  # for drawing robotic arm orientations from RK4
COM_COLOR  = (225,   0, 255)  # for drawing robotic arm CoM positions from RK4
PIN_COLOR  = (255,   0,   0)  # for drawing robotic arm pin positions from RK4

Z_VEC = (0.0,0.0,0.0)                                # zero vector
I_MAT = ((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))  # identity matrix

#-----------------------------------------------------------------------------
# Utility functions

def coord(Wxyz, integer=False):
    """
    Converts world (Wx,Wy,Wz) coordinates to screen (xs,ys) coordinates.  
    Setting 'integer' to True will return integer coordinates.
    """
    xs = (DRAW_OFFSET[0] + DRAW_SCALE*Wxyz[0])
    ys = (DRAW_OFFSET[1] - DRAW_SCALE*Wxyz[1])

    if integer:
        return int(round(xs)), int(round(ys))
    return xs, ys
        
def distAngleToXYZ(dist, ang):
    """ 
    Converts given distance and angle (in radians) to xyz coordinate.
    """
    xyz = (dist*cos(ang), dist*sin(ang), 0.0)
    
    return xyz

from tutorial2util import printVec, printFeedback

#=============================================================================

# Planar 2R Robotic Arm System - characterization and initial conditions

""" 
Diagram of the planar 2R robotic arm system.
 
                                        . 
                                       . \
                     +Y Axis     /\   .  (-T2) <--{rod 2 angle  
                        ^       /  \ .    |
                        |      /    o----------+----------o <--{tip of rod 2 is
        length of rod}--------/    / \          \               end-effector
                        |    /    /   \          --{2nd rod center of mass (CoM)
                        |   /    /     \   
                        |  /    /       --{joint 2
                        | / /\ /
        height of rod}-----/  + <--{1st rod center of mass (CoM)
                        / /  /
                       /|/  / 
                      / /  / \
                      \/| /  (+T1) <--{rod 1 angle
                       \|/    |
            ------------o------------> +X Axis
                        ^\
                        : --{joint 1
                        :
                        Origin of inertial reference system

    Notes: (1) ODE body 1 corresponding to rod 1 is positioned wrt to the 
               origin and the initial angle of rod 1 is set to Theta0_1, 
               but the angle of ODE joint 1, which attaches body 1 to the 
               origin, is always initialized to zero by ODE. Consequently, 
               Theta0_1 must be added to the values returned by the ODE 
               getAngle() method for joint 1 in order to match angle T1 
               depicted above.
             
           (2) ODE body 2 corresponding to rod 2 is positioned wrt to body
               1 and the initial angle of rod 2 is set to Theta0_2, but the
               angle of ODE joint 2, which attaches body 2 to body 1, is 
               always initialized to zero by ODE. Consequently, Theta0_2
               must be added to the values returned by the ODE getAngle() 
               method for joint 2 in order to match angle T2 depicted above. 
               Also, the sum of Theta0_1 and the getAngle value for joint 1 
               must be added to the sum of Theta0_2 and the getAngle value 
               for joint 2 in order obtain the absolute angle from the +X 
               axis for joint 2 rotations.
"""
ORIGIN  = (0.0, 0.0, 0.0)    # origin of inertial reference frame
Z_AXIS  = (0.0, 0.0, 1.0)    # axis of joint and body rotations
GRAVITY = (0.0, -9.81, 0.0)  # gravitational acceleration (m/sec/sec)

# Physical specifications.

ROD_LEN1  = 1.0                                  # length (m)
ROD_HGT1  = ROD_LEN1/2.0                         # height (m)
ROD_WID1  = 0.05                                 # width (m)
ROD_THK1  = 0.05                                 # thickness (m)
ROD_VOL1  = ROD_LEN1*ROD_WID1*ROD_THK1           # volume (m^3)
ROD_DEN1  = 7806                                 # density (kg/m^3)
ROD_M1    = ROD_DEN1*ROD_VOL1                    # mass (kg)
ROD_LSQ1  = ROD_LEN1*ROD_LEN1                    # length squared (m^2)
ROD_WSQ1  = ROD_WID1*ROD_WID1                    # width squared (m^2)
ROD_TSQ1  = ROD_THK1*ROD_THK1                    # thickness squared (m^2)
ROD_Ixxc1 = (ROD_M1/12)*(ROD_WSQ1 +   ROD_TSQ1)  # Ixx moment of inertia about CoM
ROD_Iyyc1 = (ROD_M1/12)*(ROD_TSQ1 +   ROD_LSQ1)  # Iyy moment of inertia about CoM
ROD_Izzc1 = (ROD_M1/12)*(ROD_WSQ1 +   ROD_LSQ1)  # Izz moment of inertia about CoM
ROD_Izzj1 = (ROD_M1/12)*(ROD_WSQ1 + 4*ROD_LSQ1)  # Izz moment of inertia about joint

ROD_Icom1 = ((ROD_Ixxc1,0,0), (0,ROD_Iyyc1,0), (0,0,ROD_Izzc1))
ROD_Icor1 = ((ROD_Ixxc1,0,0), (0,ROD_Iyyc1,0), (0,0,ROD_Izzj1))
ROD_Iinv1 = ((1/ROD_Ixxc1,0,0), (0,1/ROD_Iyyc1,0), (0,0,1/ROD_Izzj1))

ROD_LEN2  = 0.5                                  # length (m)
ROD_HGT2  = ROD_LEN2/2.0                         # height (m)
ROD_WID2  = 0.05                                 # width (m)
ROD_THK2  = 0.05                                 # thickness (m)
ROD_VOL2  = ROD_LEN2*ROD_WID2*ROD_THK2           # volume (m^3)
ROD_DEN2  = 7806                                 # density (kg/m^3)
ROD_M2    = ROD_DEN2*ROD_VOL2                    # mass (kg)
ROD_LSQ2  = ROD_LEN2*ROD_LEN2                    # length squared (m^2)
ROD_WSQ2  = ROD_WID2*ROD_WID2                    # width squared (m^2)
ROD_TSQ2  = ROD_THK2*ROD_THK2                    # thickness squared (m^2)
ROD_Ixxc2 = (ROD_M2/12)*(ROD_WSQ2 +   ROD_TSQ2)  # Ixx moment of inertia about CoM
ROD_Iyyc2 = (ROD_M2/12)*(ROD_TSQ2 +   ROD_LSQ2)  # Iyy moment of inertia about CoM
ROD_Izzc2 = (ROD_M2/12)*(ROD_WSQ2 +   ROD_LSQ2)  # Izz moment of inertia about CoM
ROD_Izzj2 = (ROD_M2/12)*(ROD_WSQ2 + 4*ROD_LSQ2)  # Izz moment of inertia about joint

ROD_Icom2 = ((ROD_Ixxc2,0,0), (0,ROD_Iyyc2,0), (0,0,ROD_Izzc2))
ROD_Icor2 = ((ROD_Ixxc2,0,0), (0,ROD_Iyyc2,0), (0,0,ROD_Izzj2))
ROD_Iinv2 = ((1/ROD_Ixxc2,0,0), (0,1/ROD_Iyyc2,0), (0,0,1/ROD_Izzj2))

# Physical configuration

Theta0_1 = 10.0*RPD  # initial angle of 1st rod attached to origin
Theta0_2 = 90.0*RPD  # initial angle of 2nd rod attached to 1st rod

JOINT1_ANCHOR = ORIGIN
JOINT1_AXIS   = Z_AXIS
JOINT2_ANCHOR = vecAdd(JOINT1_ANCHOR,distAngleToXYZ(ROD_LEN1,Theta0_1))
JOINT2_AXIS   = Z_AXIS

ROD1_POS = vecAdd(JOINT1_ANCHOR,distAngleToXYZ(ROD_HGT1,Theta0_1))
ROD2_POS = vecAdd(JOINT2_ANCHOR,distAngleToXYZ(ROD_HGT2,Theta0_1+Theta0_2))
EEFF_POS = vecAdd(JOINT2_ANCHOR,distAngleToXYZ(ROD_LEN2,Theta0_1+Theta0_2))

#-----------------------------------------------------------------------------
# System of ordinary differential equations (ode) characterizing the motion
# of planar 2R robotic arm comprised of two rods as described on pages 92-93
# of reference [1]. Specifically, the state space form of the differential
# equations of motion is given as
#
#   {tau} = [M(q)]*{d(dq/dt)/dt} + {V(q,dq/dt)} + {G(q)}
#
# The state variables (theta1, theta2) and associated state space equations 
# for their time derivatives (thdot1, thdot2, thddot1, thddot2) comprise 
# the "State Space" model.
#
#        State Space Model
#   ----------------------------  
#    dS[1] = d(theta1)/dt
#    dS[2] = d(theta2)/dt
#    dS[3] = d(d(theta1)/dt)/dt
#    dS[4] = d(d(theta2)/dt)/dt
#
# The dotS function incorporates the sets of first order differential 
# equations of motion. A Runge-Kutta 4th order integration method is 
# applied to the dotS function to develop the motion of the robotic
# arm in terms of the state variables S[] by solving for S[] at each 
# time step h from t0 to tStop using the following algorithm
#
#   S[](t+h) = S[](t) + ((K1[] + 2*(K2[] + K3[]) + K4[])/6)*h
#
# where
#
#   K1[] = dotS(S[](t))
#   K2[] = dotS(S[](t) + K1[]*h/2)
#   K3[] = dotS(S[](t) + K2[]*h/2)
#   K4[] = dotS(S[](t) + K3[]*h)
#
# References:
#
#   [1] Dr. Robert L. William II, "Robot Mechanics: Notesbook Supplement
#       for ME 4290/5290 Mechanics and Control of Robotic Manipulators",
#       Ohio University, Mech. Engineering, Spring 2015; web available at
#       https://www.ohio.edu/mechanical-faculty/williams/html/PDF/Supplement4290.pdf
#
#   [2] John J. Craig, "Introduction to Robotics: Mechanics and Control",
#       3rd ed., Pearson Prentice Hall, Pearson Education, Inc., Upper 
#       Saddle River, NJ, 2005

# Differential equations of motion constants.

Theta1 = Theta0_1  # initial angle of joint 1 (rad)
Theta2 = Theta0_2  # initial angle of joint 2 (rad)
tStep  = T_STEP    # simulation and integration time step (sec)
tStop  = T_STOP    # simulation and integration stop time (sec)

Xrate = np.array([0.0,0.5])  # constant commanded end-effector linear rate (m/sec)
g     = vecMag(GRAVITY)      # gravitational acceleration magnitude (m/sec/sec)

def calcJ(th1,th2):
    """
    Calculate Jacobian (pg 79 of reference [1]).
    """
    global ROD_LEN1, ROD_LEN2
    
    L1    = ROD_LEN1
    L2    = ROD_LEN2
    th12  = th1 + th2
    cth1  = cos(th1)
    sth1  = sin(th1)
    cth12 = cos(th12)
    sth12 = sin(th12)
    
    J      = np.zeros((2,2),dtype=np.float)
    J[0,0] = -L1*sth1 - L2*sth12
    J[0,1] = -L2*sth12
    J[1,0] =  L1*cth1 + L2*cth12
    J[1,1] =  L2*cth12
    
    return J
    
def calcJdot(th1,th2,thdot1,thdot2):
    """
    Calculate Jacobian time derivative (pg 79 of reference [1]).
    """
    global ROD_LEN1, ROD_LEN2
    
    L1      = ROD_LEN1
    L2      = ROD_LEN2
    th12    = th1 + th2
    thdot12 = thdot1 + thdot2
    cth1    = cos(th1)
    sth1    = sin(th1)
    cth12   = cos(th12)
    sth12   = sin(th12)
    
    Jdot      = np.zeros((2,2),dtype=np.float)
    Jdot[0,0] = -L1*cth1*thdot1 - L2*cth12*thdot12
    Jdot[0,1] = -L2*cth12*thdot12
    Jdot[1,0] = -L1*sth1*thdot1 - L2*sth12*thdot12
    Jdot[1,1] = -L2*sth12*thdot12
    
    return Jdot
    
def calcAngRate(Xrate,th1,th2):
    """
    Calculate angular rates from commanded linear rate and initial 
    joint angles (pg 79 of reference [1]). Assumes Jacobian matrix 
    J is square.
    """
    J  = calcJ(th1,th2)
    if np.linalg.matrix_rank(J) == J.ndim:
        dq = la.solve(J,np.transpose(Xrate))
    else:
        dq = la.lstsq(J,np.transpose(Xrate))
    
    return dq
    
# Initial values for state variables array

nSvar = 5                                 # number of state variables
S     = np.zeros(nSvar)                   # state variables
dS    = np.zeros(nSvar)                   # state derivatives
dq    = calcAngRate(Xrate,Theta1,Theta2)  # joint angular rates

S[0] = 0.0     # t0 
S[1] = Theta1  # joint 1 angle
S[2] = Theta2  # joint 2 angle
S[3] = dq[0]   # joint 1 angular rate
S[4] = dq[1]   # joint 2 angular rate

def dotS(n,S):
    """
    State derivatives function.
    """
    global Xrate
    
    # Use commanded rate and J to solve for angular rates.
    th1 = S[1]
    th2 = S[2]
    J   = calcJ(th1,th2)
    dq  = la.solve(J,np.transpose(Xrate))

    # Use angular rates, J and Jdot to solve for angular 
    # accelerations.
    Jdot = calcJdot(th1,th2,dq[0],dq[1])
    ddq  = la.solve(J,np.dot(-Jdot,dq))
    
    dS    = np.zeros(n) 
    dS[0] = 1.0     # d(t)/dt
    dS[1] = dq[0]   # d(th1)/dt
    dS[2] = dq[1]   # d(th2)/dt
    dS[3] = ddq[0]  # d(thdot1)/dt
    dS[4] = ddq[1]  # d(thdot2)/dt
        
    return dS

def calcStateSpace(q,dq):
    """
    Calculate matrix [M(q)], vectors {V(q,dq/dt)} and {G(q)} for the 
    the state-space representation of a planar 2R robotic arm system
    given by equations on page 93 of reference [1].
    """
    global g
    global ROD_M1, ROD_Izzc1, ROD_LEN1
    global ROD_M2, ROD_Izzc2, ROD_LEN2
    
    th1     = q[0]
    th2     = q[1]
    Dth1    = dq[0]
    Dth2    = dq[1]
    costh1  = cos(th1)
    costh2  = cos(th2)
    sinth2  = sin(th2)
    costh12 = cos(th1+th2)
    
    m1   = ROD_M1
    Izz1 = ROD_Izzc1
    L1   = ROD_LEN1
    L1sq = L1*L1
    m2   = ROD_M2
    Izz2 = ROD_Izzc2
    L2   = ROD_LEN2
    L2sq = L2*L2
    L1L2 = L1*L2
    
    M      = np.zeros((2,2),dtype=np.float)
    term1  = Izz2 + m2*L2sq/4
    term2  = m2*L1L2*costh2
    term3  = term1 + term2/2
    M[0,0] = term1 + term2 + Izz1 + (m1/4 + m2)*L1sq
    M[0,1] = term3
    M[1,0] = term3
    M[1,1] = term1
    
    V     = np.array([0,0],dtype=np.float)
    term1 = m2*L1L2*sinth2
    term2 = term1/2
    V[0]  = -(term2*Dth2 + term1*Dth1)*Dth2
    V[1]  = term2*Dth1**2
    
    G = np.array([0,0],dtype=np.float)
    term1 = g*L1*costh1
    term2 = g*m2*L2*costh12/2
    G[0]  = m1*term1/2 + m2*term1 + term2
    G[1]  = term2
    
    return (M,V,G)
    
def calcTorque(n,S):
    """
    Solves {tau} = [M(q)]*{d(dq/dt)/dt} + {V(q,dq/dt)} + {G(q)}.
    """
    dS      = dotS(n,S)
    q       = (S[1],S[2])
    dq      = (dS[1],dS[2])
    ddq     = np.array([dS[3], dS[4]],dtype=np.float)
    (M,V,G) = calcStateSpace(q,dq)
    tau     = np.dot(M,np.transpose(ddq)) + np.transpose(V+G)
    
    return tau
    
def calcAngAccel(Tq,q,dq):
    """
    Solves [M(q)]*{d(dq/dt)/dt} = {tau} - {V(q,dq/dt)} - {G(q)}
    for {d(dq/dt)/dt}. Assumes mass matrix M is square.
    """
    tau     = np.array([Tq[0],Tq[1]],dtype=np.float)
    (M,V,G) = calcStateSpace(q,dq)
    b       = np.transpose(tau-V-G)
    if np.linalg.matrix_rank(M) == M.ndim:
        ddq = la.solve(M,b,sym_pos=True)
    else:
        ddq = la.lstsq(M,b)
    
    return ddq

def calcLinAccel12(S,dS):
    """
    Calculates linear acceleration at center of mass for rods 1 and 2
    using equations (8.11) and (8.12) on page 275 of reference [3].
    """
    global ROD_LEN1
    global ROD_LEN2
    
    L1 = ROD_LEN1
    hL1 = L1/2
    L2 = ROD_LEN2
    hL2 = L2/2
    
    th1 = S[1]
    th2 = S[2]
    th12 = th1 + th2
    c1 = cos(th1)
    s1 = sin(th1)
    c12 = cos(th12)
    s12 = sin(th12)
    
    thdot1 = dS[1]
    thdot2 = dS[2]
    thdot12 = thdot1 + thdot2
    thdot1sq = thdot1*thdot1
    thdot12sq = thdot12*thdot12
    thddot1 = dS[3]
    thddot2 = dS[4]
    thddot12 = thddot1 + thddot2

    a1x = -hL1*thdot1sq*c1 - hL1*thddot1*s1
    a1y = -hL1*thdot1sq*s1 + hL1*thddot1*c1
    a1z = 0.0
    a1 = np.array([a1x, a1y, a1z], dtype=np.float)
    
    a2x = -L1*thdot1sq*c1 - hL2*thdot12sq*c12 - L1*thddot1*s1 - hL2*thddot12*s12
    a2y = -L1*thdot1sq*s1 - hL2*thdot12sq*s12 + L1*thddot1*c1 + hL2*thddot12*c12
    a2z = 0.0
    a2 = np.array([a2x, a2y, a2z], dtype=np.float)
    
    return a1, a2
    
def calcLinAccelEE(n,S):
    """
    Calculates linear acceleration of end-effector using equations
    on pages 79 of reference [1].
    """
    th1    = S[1]
    th2    = S[2]
    dS     = dotS(n,S)
    thdot1 = dS[1]
    thdot2 = dS[2]
    J      = calcJ(th1,th2)
    Jdot   = calcJdot(th1,th2,thdot1,thdot2)
    dq     = np.array([dS[1], dS[2]],dtype=np.float)
    ddq    = np.array([dS[3], dS[4]],dtype=np.float)
    a      = np.dot(J,np.transpose(ddq)) + np.dot(Jdot,np.transpose(dq))
    
    return (a[0], a[1], 0.0)
    
def calcInvDynamicsRK4(jList,S,dS):
    """
    Calculates inverse dynamics for RK4 model from given joint list 
    with associated state variables and derivatives using Recursive
    Newton-Euler Algorithm. See pages 76-78 and 84 of reference [1]
    and pages 177-180 of reference [2].
    """
    global Z_VEC, I_MAT
    global ORIGIN
    global ROD_LEN1, ROD_HGT1, ROD_M1, ROD_Icom1
    global ROD_LEN2, ROD_HGT2, ROD_M2, ROD_Icom2
    global g
        
    m  = [0.0, ROD_M1, ROD_M2]
    I  = [I_MAT, ROD_Icom1, ROD_Icom2]
    Ag = (0.0, g, 0.0)  # gravitational acceleration offset
    
    if PRINT_EVAL:
        print('calcInvDynamicsRK4:')
        
    # Points of interest locations in local joint frame coordinates.
    # Note: Element 0 corresponds to joint/body 0, which is the 
    #       virtual robotic base fixed at the origin of the ODE 
    #       environment, and element 3 corresponds to the end-
    #       effector (a virtual joint/body) located at the tip 
    #       of rod 2.
    # joint axes locations
    Pja = [ORIGIN,ORIGIN,(ROD_LEN1,0.0,0.0),(ROD_LEN2,0.0,0.0)]
    # body CoM locations wrt to inboard joint axis location
    Pcg = [ORIGIN,(ROD_HGT1,0.0,0.0),(ROD_HGT2,0.0,0.0),(ROD_LEN2,0.0,0.0)]
    
    # Recursion work space variables (measures in global reference frame).
    # Note: The following variable lists contain a zero scalar or vector 
    #       (Z_VEC) in the 0th entry to simplify outbound and inbound 
    #       recursion computations below. Also, the 0th entry in state 
    #       variables S and dS correspond to t and dt respectively, thus
    #       the rotation angle and rates for joints 1 and 2 occupy the 
    #       1st and 2nd entries of S for angle, and the 1st/3rd and 
    #       2nd/4th entries of dS for angular velocity/acceleration.
    rth = [0.0,0.0,0.0,0.0]   # joint relative rotation angles
    ath = [0.0,0.0,0.0,0.0]   # joint absolute rotation angles
    Pj  = [Z_VEC,Z_VEC,Z_VEC]  # joint position vectors
    Vj  = [Z_VEC,Z_VEC,Z_VEC]  # joint linear velocity vectors
    Aj  = [Z_VEC,Z_VEC,Z_VEC]  # joint linear acceleration vectors
    w   = [Z_VEC,Z_VEC,Z_VEC]  # joint/body angular velocity vectors
    a   = [Z_VEC,Z_VEC,Z_VEC]  # joint/body angular acceleration vectors
    Pb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body position vectors
    Vb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body linear velocity vectors
    Ab  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body linear acceleration vectors
    Fb  = [Z_VEC,Z_VEC,Z_VEC]        # inertial forces (in joint/body frame)
    Nb  = [Z_VEC,Z_VEC,Z_VEC]        # inertial moments (in joint/body frame)
    fb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # internal forces (in joint/body frame)
    nb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # total torque at inboard joint
    nbo = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # moment about body cg from outboard joint
    nbi = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # nb minus nbo
    tau = [0.0,0.0,0.0]              # total torque on inboard joint

    # Outbound joint list traversal for joint/body positions, orientations, 
    # velocities and accelerations, inertial forces and moments.
    if PRINT_EVAL:
        print('...outbound kinematics iteration.')
    nj = len(jList)
    kj = nj - 1
    ij = 1
    for j in jList[ij:nj]:
        # ... ith joint rotation angles and rates
        rth[ij] = S[ij]
        ath[ij] = ath[ij-1] + rth[ij]
        Zdth    = vecMulS(Z_AXIS,dS[ij]) 
        alpha   = vecMulS(Z_AXIS,dS[ij+kj])
        # ... joint and body pose positions
        Pj[ij] = vecAdd(Pj[ij-1],vecRotZ(Pja[ij],-ath[ij-1]))
        Pb[ij] = vecAdd(Pj[ij],vecRotZ(Pcg[ij],-ath[ij]))
        # ... get vectors from last joint to this joint (D0), and
        #     from this joint to its body COM (C1)
        D0 = vecSub(Pj[ij],Pj[ij-1])
        C1 = vecSub(Pb[ij],Pj[ij])
        # ... joint angular velocity and acceleration, eqns (6.45 & 6.46) Ref [2]
        w[ij] = vecAdd(w[ij-1],Zdth)
        a[ij] = vecAdd(a[ij-1],vecAdd(vecCrossP(w[ij-1],Zdth),alpha))
        # ... joint velocity and acceleration, eqns (5.47 & 6.47) Ref [2]
        Vj[ij] = vecAdd(Vj[ij-1],vecCrossP(w[ij-1],D0))
        Aj[ij] = vecAdd(Aj[ij-1],\
                        vecAdd(vecCrossP(a[ij-1],D0),\
                               vecCrossP(w[ij-1],\
                                         vecCrossP(w[ij-1],D0))))
        # ... body velocity and acceleration, eqn (6.48) Ref [2]
        Vb[ij] = vecAdd(Vj[ij],vecCrossP(w[ij],C1))
        Ab[ij] = vecAdd(Aj[ij],\
                        vecAdd(vecCrossP(a[ij],C1),\
                               vecCrossP(w[ij],\
                                         vecCrossP(w[ij],C1))))
        # ... inertial loading (kinetics)
        Fb[ij] = vecMulS(vecRotZ(vecAdd(Ab[ij],Ag),ath[ij]),m[ij])
        Nb[ij] = vecAdd(matDotV(I[ij],a[ij]),\
                        vecCrossP(w[ij],matDotV(I[ij],w[ij])))
        if PRINT_EVAL:
            printVec('j'+str(ij), 'w', w[ij])
            printVec('j'+str(ij), 'a', a[ij])
            printVec('j'+str(ij), 'Vj', Vj[ij])
            printVec('j'+str(ij), 'Aj', Aj[ij])
            printVec('b'+str(ij), 'Vb', Vb[ij])
            printVec('b'+str(ij), 'Wb', w[ij])  # must be same as joint w[ij]
            printVec('b'+str(ij), 'Ab', Ab[ij])
            printVec('b'+str(ij), 'Fb', Fb[ij])
            printVec('b'+str(ij), 'Nb', Nb[ij])
        ij = ij + 1

    # End-effector position, velocity and acceleration.
    Pb[ij] = vecAdd(Pj[ij-1],vecRotZ(Pcg[ij],-ath[ij-1]))
    Vb[ij] = vecAdd(Vj[ij-1],vecCrossP(w[ij-1],vecSub(Pb[ij],Pj[ij-1])))
    Ab[ij] = vecAdd(Aj[ij-1],\
                    vecAdd(vecCrossP(a[ij-1],\
                                     vecSub(Pb[ij],Pj[ij-1])),\
                           vecCrossP(w[ij-1],\
                                     vecCrossP(w[ij-1],\
                                               vecSub(Pb[ij],Pj[ij-1])))))
    if PRINT_EVAL:
        printVec('ee', 'Vb', Vb[ij])
        printVec('ee', 'Ab', Ab[ij])
    
    # Inbound iteration over reverse of joint list.
    if PRINT_EVAL:
        print('...inbound kinetics iteration.')
    jList.reverse()
    ij = kj
    for j in jList[0:kj]:
        # calculate fb in local joint/body frames, but display in global 
        # reference frame coordinates for comparision to feedback forces.
        fb[ij] = vecAdd(vecRotZ(fb[ij+1],-rth[ij+1]),Fb[ij])
        nb[ij] = vecAdd(vecRotZ(nb[ij+1],-rth[ij+1]),\
                        vecAdd(vecCrossP(Pcg[ij],Fb[ij]),\
                               vecAdd(vecCrossP(Pja[ij+1],\
                                                vecRotZ(fb[ij+1],-rth[ij+1])),\
                                      Nb[ij])))
        tau[ij] = nb[ij][2]                
        # these inboard (nbi) and outboard (nbo) moments are for comparison 
        # to feedback torque fb[1] for joint[ij] and fb[3] for joint[ij+1].
        nbo[ij] = vecAdd(vecRotZ(nb[ij+1],-rth[ij+1]),\
                         vecCrossP(vecSub(Pcg[ij],Pja[ij]),\
                                   vecRotZ(fb[ij+1],-rth[ij+1])))
        nbi[ij] = vecSub(nb[ij],nbo[ij])
        nbo[ij] = vecSub(nbo[ij],nbi[ij+1])
        if PRINT_EVAL:
            printVec('j'+str(ij), 'fb', vecRotZ(fb[ij],-ath[ij]))
            printVec('j'+str(ij), 'nb', nb[ij])
            printVec('j'+str(ij), 'nbi', nbi[ij])
            printVec('j'+str(ij), 'nbo', nbo[ij])
        ij = ij - 1
        
    if PRINT_EVAL:
        print('...return.')
        
    return (Pj,rth,ath,w,a,Pb,Vb,Ab,(tau[1],tau[2]))

def calcFwdDynamicsODE(jList):
    """
    Calculates forward dynamics for ODE model from given joint list
    using Articulated-Body Algorithm. Uses forces and torques from 
    joint feedback data to compute linear and angular accelerations
    which are not provided by ODE. See pages 76-78 and 84 of ref [1]
    and pages 177-180 of ref [2].
    """
    global body1, body2, j2
    global Theta0_1, Theta0_2
    global I_MAT, ROD_HGT2
    global g
    
    th0 = [0.0,Theta0_1,Theta0_2]  # initial joint angles
    Ag  = (0.0, g, 0.0)            # gravitational acceleration offset
   
    if PRINT_EVAL:
        print('calcFwdDynamicsODE:')
        
    # Recursion work space variables (measures in global reference frame).
    # Note: The following variable lists contain a zero scalar or vector 
    #       (Z_VEC) in the 0th entry to simplify outbound and inbound 
    #       recursion computations below.
    rth = [0.0,0.0,0.0,0.0]   # joint relative rotation angles
    ath = [0.0,0.0,0.0,0.0]   # joint absolute rotation angles
    Pj  = [Z_VEC,Z_VEC,Z_VEC]  # joint position vectors
    Vj  = [Z_VEC,Z_VEC,Z_VEC]  # joint linear velocity vectors
    Aj  = [Z_VEC,Z_VEC,Z_VEC]  # joint linear acceleration vectors
    w   = [Z_VEC,Z_VEC,Z_VEC]  # joint/body angular velocity vectors
    a   = [Z_VEC,Z_VEC,Z_VEC]  # joint/body angular acceleration vectors
    Pb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body position vectors
    Vb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body linear velocity vectors
    Ab  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body linear acceleration vectors
    Wb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # body angular velocity vectors
    Fb  = [Z_VEC,Z_VEC,Z_VEC]        # inertial forces (in joint/body frame)
    Nb  = [Z_VEC,Z_VEC,Z_VEC]        # inertial moments (in joint/body frame)
    fb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # internal forces (in joint/body frame)
    nb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # total torque at inboard joint
    nbo = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # moment about body cg from force at outboard joint
    nbi = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # moment about inboard joint from force at body cg
    tau = [0.0,0.0,0.0]              # total torque on inboard joint
    qdd = [Z_VEC,Z_VEC,Z_VEC]        # inboard joint's generalized angular acceleration

    # Outbound kinematics iteration.
    if PRINT_EVAL:
        print('...outbound kinematics iteration.')
    nj = len(jList)
    kj = nj - 1
    ij = 1
    for j in jList[ij:nj]:
        # joint body mass, moments of inertia and rotation matrices
        b  = j.getBody(0)
        bM = b.getMass()
        m  = bM.mass
        I  = bM.I
        # ... joint rotation angle and rate
        rth[ij] = th0[ij] + j.getAngle()  # add initial offset (see diagram notes)
        ath[ij] = ath[ij-1] + rth[ij]
        Zdth    = vecMulS(j.getAxis(),j.getAngleRate())
        # ... joint and body pose positions
        Pj[ij] = j.getAnchor()
        Pb[ij] = b.getPosition()
        # ... get vectors from last joint to this joint (D0), and
        #     from this joint to its body COM (C1)
        D0 = vecSub(Pj[ij],Pj[ij-1])
        C1 = vecSub(Pb[ij],Pj[ij])
        # ... joint inertial angular velocity
        w[ij] = vecAdd(w[ij-1],Zdth)
        # ... joint inertial linear velocity
        Vj[ij] = vecAdd(Vj[ij-1],vecCrossP(w[ij-1],D0))
        # ... joint inertial linear acceleration
        Aj[ij] = vecAdd(Aj[ij-1],\
                        vecAdd(vecCrossP(a[ij-1],D0),\
                               vecCrossP(w[ij-1],vecCrossP(w[ij-1],D0))))
        # ... joint inertial angular acceleration (from joint feedback) by
        #     solving Ftot = m*(a[ij]xC1 + w[ij]x(w[ij]xC1) + Aj[ij]) for 
        #     a[ij]
        fbck1 = j.getFeedback()  # for this joint's force on body
        if PRINT_FBCK: printVec('j'+str(ij), 'fbck1[0]', fbck1[0])
        if ij < kj:
            # account for next joint's force on body
            fbck2 = jList[ij+1].getFeedback()
            if PRINT_FBCK: printVec('j'+str(ij), 'fbck2[2]', fbck2[2])
            fbtot = vecAdd(fbck1[0],fbck2[2])
        else:
            fbtot = fbck1[0]
        if PRINT_FBCK: printVec('j'+str(ij), 'fbtot', fbtot)
        # note: remove gravitational offset applied by ODE
        a[ij] = vecSub(vecSub(vecDivS(fbtot,m),Ag),\
                       vecAdd(vecCrossP(w[ij],vecCrossP(w[ij],C1)),Aj[ij]))
        a[ij] = matDotV(la.pinv(transposeM(vecToSkewMat(C1))),a[ij])
        qdd[ij] = vecSub(a[ij],a[ij-1]) 
        # ... body inertial linear and angular velocity
        Vb[ij] = b.getLinearVel()
        Wb[ij] = b.getAngularVel()
        # ... body inertial linear acceleration
        Ab[ij] = vecAdd(Aj[ij],\
                        vecAdd(vecCrossP(a[ij],C1),\
                               vecCrossP(w[ij],vecCrossP(w[ij],C1))))
        # ... inertial loading (kinetics)
        Fb[ij] = vecMulS(vecAdd(Ab[ij],Ag),m)  # reapply gravitational offset
        Nb[ij] = vecAdd(matDotV(I,a[ij]),\
                        vecCrossP(w[ij],matDotV(I,w[ij])))
        if PRINT_EVAL:
            printVec('j'+str(ij), 'w', w[ij])
            printVec('j'+str(ij), 'a', a[ij])
            printVec('j'+str(ij), 'qdd', qdd[ij])
            printVec('j'+str(ij), 'Vj', Vj[ij])
            printVec('j'+str(ij), 'Aj', Aj[ij])
            printVec('b'+str(ij), 'Vb', Vb[ij])
            printVec('b'+str(ij), 'Wb', Wb[ij])  # must be same as joint w[ij]
            printVec('b'+str(ij), 'Ab', Ab[ij])
            printVec('b'+str(ij), 'Fb', Fb[ij])
            printVec('b'+str(ij), 'Nb', Nb[ij])   
        ij = ij + 1
        
    # End-effector position and velocity.
    Pb[ij] = jList[nj-1].getBody(0).getRelPointPos((ROD_HGT2,0.0,0.0))
    Vb[ij] = jList[nj-1].getBody(0).getRelPointVel((ROD_HGT2,0.0,0.0))
    Ab[ij] = vecAdd(Aj[ij-1],\
                    vecAdd(vecCrossP(a[ij-1],\
                                     vecSub(Pb[ij],Pj[ij-1])),\
                           vecCrossP(w[ij-1],\
                                     vecCrossP(w[ij-1],\
                                               vecSub(Pb[ij],Pj[ij-1])))))
    if PRINT_EVAL:
        printVec('ee', 'Vb', Vb[ij])
        printVec('ee', 'Ab', Ab[ij])
        
    # Inbound articulated body inertia and bias force recursion.
    if PRINT_EVAL:
        print('...inbound kinetics iteration.')
    jList.reverse()
    ij = kj
    for j in jList[0:kj]:
        # joint body mass, moments of inertia and rotation matrices
        b  = j.getBody(0)
        bM = b.getMass()
        m  = bM.mass
        I  = bM.I
        # body CoM and joint relative location vectors in global frame
        C1 = vecSub(Pb[ij],Pj[ij])
        # these inboard (nbi) and outboard (nbo) moments are for comparison 
        # to feedback torque fb[1] for joint[ij] and fb[3] for joint[ij+1]
        if ij < kj:  # contributions from inboard and outboard joints
            D0 = vecSub(Pj[ij+1],Pj[ij])
            fj = vecAdd(fb[ij+1],Fb[ij])
            mj = vecAdd(vecAdd(vecAdd(Nb[ij],nb[ij+1]),\
                               vecCrossP(C1,Fb[ij])),\
                        vecCrossP(D0,fb[ij+1]))
            nbo[ij] = vecCrossP(vecSub(Pb[ij],Pj[ij+1]),fb[ij+1])
            nbi[ij] = vecAdd(mj,vecSub(nbo[ij],nbi[ij+1]))
        else:        # contributions from inboard joint only
            fj = Fb[ij]
            mj = vecAdd(Nb[ij],vecCrossP(C1,Fb[ij]))
            nbi[ij] = vecCrossP(C1,Fb[ij])
            nbo[ij] = Z_VEC
        fb[ij]  = fj
        nb[ij]  = mj
        tau[ij] = nb[ij][2]
        if PRINT_EVAL:
            printVec('j'+str(ij), 'fb', fb[ij])
            printVec('j'+str(ij), 'nb', nb[ij])
            printVec('j'+str(ij), 'nbi', nbi[ij])
            printVec('j'+str(ij), 'nbo', nbo[ij])
        ij = ij - 1
        
    if PRINT_EVAL:
        print('...return.')
        
    return (Pj,rth,ath,w,a,Pb,Vb,Wb,Ab,(qdd[1][2],qdd[2][2]),(tau[1],tau[2]))

#=============================================================================
    
# Initialize pygame
pygame.init()

# Open a display
screen = pygame.display.set_mode(WINDOW_RESOLUTION)

# Create an ODE world object
world = ode.World()
world.setGravity(GRAVITY)
world.setERP(1.0)
world.setCFM(1.0E-6)

# Create fixed base for robotic arm
body0 = ode.Body(world)
mass0 = ode.Mass()
mass0.setSphereTotal(50.0,ROD_WID1)
body0.setMass(mass0)
body0.setPosition(ORIGIN)

# Create two bodies in rest position (i.e., joint/motor angles at zero)
body1 = ode.Body(world)
mass1 = ode.Mass()
mass1.setBoxTotal(ROD_M1, ROD_LEN1, ROD_WID1, ROD_THK1)
body1.setMass(mass1)
body1.setPosition(ROD1_POS)
body1.setQuaternion((cos(Theta0_1/2),0,0,sin(Theta0_1/2)))

body2 = ode.Body(world)
mass2 = ode.Mass()
mass2.setBoxTotal(ROD_M2, ROD_LEN2, ROD_WID2, ROD_THK2)
body2.setMass(mass2)
body2.setPosition(ROD2_POS)
body2.setQuaternion((cos((Theta0_1+Theta0_2)/2),0,0,sin((Theta0_1+Theta0_2)/2)))

# Connect fixed base to environment
j0 = ode.FixedJoint(world)
j0.attach(body0,ode.environment)
j0.setFixed()

# Connect body1 with fixed base
j1 = ode.HingeJoint(world)
j1.attach(body1, body0)
j1.setAnchor(JOINT1_ANCHOR)
j1.setAxis(JOINT1_AXIS)
j1.setFeedback(True)

# Connect body2 with body1
j2 = ode.HingeJoint(world)
j2.attach(body2, body1)
j2.setAnchor(JOINT2_ANCHOR)
j2.setAxis(JOINT2_AXIS)
j2.setFeedback(True)

# Add axial motors to joints
am1 = ode.AMotor(world)
am1.attach(j1.getBody(0), j1.getBody(1))
am1.setMode(ode.AMotorUser)
am1.setNumAxes(1)
am1.setAxis(0, 1, JOINT1_AXIS)
am1.setAngle(0,Theta0_1)
am1.setFeedback(True)
if MOTOR_AVEL: am1.setParam(ode.ParamFMax,5000.0)

am2 = ode.AMotor(world)
am2.attach(j2.getBody(0), j2.getBody(1))
am2.setMode(ode.AMotorUser)
am2.setNumAxes(1)
am2.setAxis(0, 1, JOINT2_AXIS)
am2.setAngle(0,Theta0_2)
am2.setFeedback(True)
if MOTOR_AVEL: am2.setParam(ode.ParamFMax,1000.0)

# Define pygame circle radius for drawing each joint sphere
cir_rad = int(BOX_WIDTH)

# Define end-effector path line end points
path0 = (EEFF_POS[0], EEFF_POS[1], 0.0)
path1 = (EEFF_POS[0]+T_STOP*Xrate[0], EEFF_POS[1]+T_STOP*Xrate[1], 0.0)

# Create background for text and clearing screen
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(FILL_COLOR)

# Write title
if pygame.font:
    title   = "PyODE Tutorial 2 - Planar 2R Robotic Arm Sim"
    font    = pygame.font.Font(None, 30)
    text    = font.render(title, 1, TEXT_COLOR)
    textpos = text.get_rect(centerx=background.get_width()/2)
    background.blit(text, textpos)
    font = pygame.font.Font(None, 24)
    
# Clear screen
screen.blit(background, (0, 0))

# Simulation loop...

FPS    = 50
F_TIME = 1.0/FPS
N_STEP = int(floor((F_TIME + 0.5*T_STEP)/T_STEP))
N_TIME = N_STEP*T_STEP

if __name__ == "__main__":
    
    # Create simulation data collection arrays for plotting.
    
    nSamples = int(ceil(T_STOP/N_TIME)) + 1
    
    if PLOT_DATA:
        Time    = np.zeros(nSamples)  # simulation time
        EEXPode = np.zeros(nSamples)  # end-effector X position from ODE
        EEYPode = np.zeros(nSamples)  # end-effector Y position from ODE
        EEARode = np.zeros(nSamples)  # end-effector angle in radians from ODE
        EEXPrk4 = np.zeros(nSamples)  # end-effector X position from RK4
        EEYPrk4 = np.zeros(nSamples)  # end-effector Y position from RK4
        EEARrk4 = np.zeros(nSamples)  # end-effector angle in radians from RK4
        EEXVode = np.zeros(nSamples)  # end-effector X velocity from ODE
        EEYVode = np.zeros(nSamples)  # end-effector Y velocity from ODE
        EEXVrk4 = np.zeros(nSamples)  # end-effector X velocity from RK4
        EEYVrk4 = np.zeros(nSamples)  # end-effector Y velocity from RK4
        J1ARode = np.zeros(nSamples)  # joint 1 angle in radians from ODE
        J2ARode = np.zeros(nSamples)  # joint 2 angle in radians from ODE
        J1ADode = np.zeros(nSamples)  # joint 1 angle in degrees from ODE
        J2ADode = np.zeros(nSamples)  # joint 2 angle in degrees from ODE
        J1ADrk4 = np.zeros(nSamples)  # joint 1 angle in degrees from RK4
        J2ADrk4 = np.zeros(nSamples)  # joint 2 angle in degrees from RK4
        B1LVode = np.zeros(nSamples)  # body 1 linear velocity from ODE
        B2LVode = np.zeros(nSamples)  # body 2 linear velocity from ODE
        B1LAode = np.zeros(nSamples)  # body 1 linear acceleration from ODE
        B2LAode = np.zeros(nSamples)  # body 2 linear acceleration from ODE
        B1LVrk4 = np.zeros(nSamples)  # body 1 linear velocity from RK4
        B2LVrk4 = np.zeros(nSamples)  # body 2 linear velocity from RK4
        B1LArk4 = np.zeros(nSamples)  # body 1 linear acceleration from RK4
        B2LArk4 = np.zeros(nSamples)  # body 2 linear acceleration from RK4
        B1LAeom = np.zeros(nSamples)  # body 1 linear acceleration from EOM
        B2LAeom = np.zeros(nSamples)  # body 2 linear acceleration from EOM
        B1AVode = np.zeros(nSamples)  # body 1 angular velocity from ODE
        B2AVode = np.zeros(nSamples)  # body 2 angular velocity from ODE
        B1AVrk4 = np.zeros(nSamples)  # body 1 angular velocity from RK4
        B2AVrk4 = np.zeros(nSamples)  # body 2 angular velocity from RK4
        J1AVode = np.zeros(nSamples)  # joint 1 angular velocity from ODE
        J2AVode = np.zeros(nSamples)  # joint 2 angular velocity from ODE
        J1AVrk4 = np.zeros(nSamples)  # joint 1 angular velocity from RK4
        J2AVrk4 = np.zeros(nSamples)  # joint 2 angular velocity from RK4
        J1AAode = np.zeros(nSamples)  # joint 1 angular acceleration from ODE
        J2AAode = np.zeros(nSamples)  # joint 2 angular acceleration from ODE
        J1AArk4 = np.zeros(nSamples)  # joint 1 angular acceleration from RK4
        J2AArk4 = np.zeros(nSamples)  # joint 2 angular acceleration from RK4
        J1TQode = np.zeros(nSamples)  # joint 1 torque from ODE
        J2TQode = np.zeros(nSamples)  # joint 2 torque from ODE
        J1TQrk4 = np.zeros(nSamples)  # joint 1 torque from RK4
        J2TQrk4 = np.zeros(nSamples)  # joint 2 torque from RK4
    
    # Instantiate clock to regulate display updates. 
    
    clk = pygame.time.Clock()
    
    # Set initial linear and angular velocities of the bodies based on the
    # initial angular velocities (dq) of the robotic arm joints.
    
    j1Pos = j1.getAnchor()
    j2Pos = j2.getAnchor()
    b1    = j1.getBody(0)
    b2    = j2.getBody(0)
    b1Pos = b1.getPosition()
    b2Pos = b2.getPosition()
    
    angVel1 = vecMulS(Z_AXIS,S[3])
    angVel2 = vecMulS(Z_AXIS,S[3]+S[4])
   
    b1.setLinearVel(vecCrossP(angVel1,vecSub(b1Pos,j1Pos)))
    b1.setAngularVel(angVel1)
    linVel = vecCrossP(angVel1,vecSub(j2Pos,j1Pos))
    b2.setLinearVel(vecAdd(vecCrossP(angVel2,vecSub(b2Pos,j2Pos)),linVel))
    b2.setAngularVel(angVel2)
    
    # Set initial angular velocities of, or torques on, axial motors based
    # on the initial angular velocities (dq) of the robotic arm joints.
    
    tau = calcTorque(nSvar,S)
    
    # set ODE joint motors angular rate or torque
    if MOTOR_AVEL: am1.setParam(ode.ParamVel,S[3])
    else:          am1.addTorques(tau[0],0.0,0.0)
    if MOTOR_AVEL: am2.setParam(ode.ParamVel,S[4])
    else:          am2.addTorques(tau[1],0.0,0.0)
    
    # NOTE: Although body velocities and joint angular motor rotational
    #       velocity or torque have been initialized, forces and moments
    #       on the bodies are still zero, which will be apparent in the
    #       comparison plots of body linear and angular accelerations and
    #       joint rotational accelerations and torques between ODE and RK4.
       
    # Step ODE (to set rates and feedback data).
    
    world.step(T_STEP/100.0)
    
    # Instantiate a temporary Runge-Kutta 4th order ode solver, 
    # initialize, take same very small step to match ODE and
    # save system state from step, then delete the RK4 solver 
    # object.
    
    rk4temp = RK4_Solver(tStep/100.0,nSvar)
    
    rk4temp.init(S)
    
    S = rk4temp.step(S,dotS)
    
    del rk4temp
        
    # Instantiate Runge-Kutta 4th order ode solver, initialize
    # using current state with state time set to zero.

    rk4 = RK4_Solver(tStep,nSvar)
    
    S[0] = 0.0
    
    rk4.init(S)
    
    tau  = calcTorque(nSvar,S)
    J    = calcJ(S[1],S[2])
    dq   = calcAngRate(Xrate,S[1],S[2])
    Jdot = calcJdot(S[1],S[2],dq[0],dq[1])
    
    print('Xrate = ',Xrate)
    print('q = ',S[1],S[2])
    print('J = ',J)
    print('dq = ',dq[0],dq[1])
    print('Jdot = ',Jdot)
    print('tau = ', tau)
    
    # Loop until termination event or simulation stop condition reached.
    
    loopFlag = True
    t        = S[0]
    i        = 0
    while loopFlag and i < nSamples:
        
        # Check for loop termination event.
        for e in pygame.event.get():
            if e.type == QUIT:
                loopFlag=False
            if e.type == KEYDOWN:
                loopFlag=False
                
        # Clear the screen.
        screen.blit(background, (0, 0))
        
        # Display simulation time.
        if font:
            tformat    = "Simulation Time = %8.3f (sec)"
            text      = font.render(tformat % t, 1, TEXT_COLOR)
            textpos   = text.get_rect(centerx=background.get_width()/2)
            hoffset   = int(ceil(1.5*textpos.height))
            textpos.y = background.get_height() - hoffset
            screen.blit(text, textpos)
        
        # Get ODE joint feedback forces and torques.
        fb1 = j1.getFeedback()
        fb2 = j2.getFeedback()
        if PRINT_FBCK:
            printFeedback('j1',t,fb1)
            printFeedback('j2',t,fb2)
                    
        # Evaluate ODE joint/body kinematics and dynamics
        results_ode = calcFwdDynamicsODE([j0,j1,j2])
        (Pj,rth,ath,w_ode,a_ode,Pb,Vb,Wb,Ab,qdd_ode,Tq_ode) = results_ode
        
        j1Pos   = Pj[1]  # joint 1
        j2Pos   = Pj[2]  # joint 2
        b1Pos   = Pb[1]  # CoM of rod 1
        b1LVel  = Vb[1]
        b1AVel  = Wb[1]
        b2Pos   = Pb[2]  # CoM of rod 2
        b2LVel  = Vb[2]
        b2AVel  = Wb[2]
        j3Pos   = Pb[3]  # end-effector at
        j3Vel   = Vb[3]  #   tip of rod 2
        j1angle = rth[1]
        j2angle = rth[2]
        j1theta = ath[1]
        j2theta = ath[2] 
        j1omega = j1.getAngleRate()
        j2omega = j2.getAngleRate()
         
        if PLOT_DATA:
            # for figures 1, 2, 4 and 6
            EEXPode[i] = j3Pos[0]
            EEYPode[i] = j3Pos[1]
            EEARode[i] = j1angle + j2angle
            EEXVode[i] = j3Vel[0]
            EEYVode[i] = j3Vel[1]
            B1LAode[i] = vecMag(Ab[1])
            B2LAode[i] = vecMag(Ab[2])
            J1ADode[i] = j1angle*DPR
            J2ADode[i] = j2angle*DPR
            
        # Get RK4 model joint angle rates and accelerations.
        dS = dotS(nSvar,S)
        
        # Evaluate RK4 joint/body kinematics and dynamics.
        results_rk4 = calcInvDynamicsRK4([j0,j1,j2],S,dS)
        (Pj,rth,ath,w_rk4,a_rk4,Pb,Vb,Ab,Tq_rk4) = results_rk4
        
        # Calculate body linear accelerations using Newtonian equations of motion.
        (a1, a2) = calcLinAccel12(S,dS)
        
        p1Pos   = Pj[1]  # joint 1
        p2Pos   = Pj[2]  # joint 2
        m1Pos   = Pb[1]  # CoM of rod 1
        m1Vel   = Vb[1]  
        m2Pos   = Pb[2]  # CoM of rod 2
        m2Vel   = Vb[2]
        p3Pos   = Pb[3]  # end-effector at
        p3Vel   = Vb[3]  #   tip of rod 2
        angVel1 = w_rk4[1]
        angVel2 = w_rk4[2]
        
        if PLOT_DATA:
            # for figures 1, 2, 4 and 6
            EEXPrk4[i] = p3Pos[0]
            EEYPrk4[i] = p3Pos[1]
            EEARrk4[i] = S[1] + S[2]
            EEXVrk4[i] = p3Vel[0]
            EEYVrk4[i] = p3Vel[1]
            B1LArk4[i] = vecMag(Ab[1])
            B2LArk4[i] = vecMag(Ab[2])
            B1LAeom[i] = vecMag(a1)
            B2LAeom[i] = vecMag(a2)
            J1ADrk4[i] = S[1]*DPR
            J2ADrk4[i] = S[2]*DPR
            
        # Collect data for printing and plotting.
        # ... time steps
        if PRINT_DATA and not PRINT_FBCK:
            print("t : %7.3f" % t)
        if PLOT_DATA: Time[i] = t
        # ... linear and angular velocity for body 1 from ODE and RK4
        if PRINT_DATA:
            j_fmt = "j1:  angle= %8.3f  theta = %8.3f  omega = %8.3f"
            b_fmt = "b1:  LVel = %8.3f %8.3f %8.3f  AVel = %8.3f %8.3f %8.3f"
            print(j_fmt % (j1angle*DPR, j1theta*DPR, j1omega) )
            print(b_fmt % \
                (b1LVel[0],b1LVel[1],b1LVel[2],b1AVel[0],b1AVel[1],b1AVel[2]) )
        Velode = vecCrossP(b1AVel,vecSub(b1Pos,j1Pos))
        Velrk4 = m1Vel
        if PRINT_DATA:
            printVec('b1','Velode',Velode)  # Velode should be same as b1LVel
            printVec('b1','Velrk4',Velrk4)
            printVec('b1','b1AVel',b1AVel)  
            printVec('b1','angVel1',angVel1)
        if PLOT_DATA:
            # for figures 3 and 5
            B1LVode[i] = vecMag(b1LVel)
            B1LVrk4[i] = vecMag(Velrk4)
            B1AVode[i] = b1AVel[2]*DPR   # b1AVel[2] should be same as j1omega 
            B1AVrk4[i] = angVel1[2]*DPR
        # ... linear and angular velocity for body 2 from ODE and RK4
        if PRINT_DATA:
            j_fmt = "j2:  angle = %8.3f  theta = %8.3f  omega = %8.3f"
            b_fmt = "b2:  LVel = %8.3f %8.3f %8.3f  AVel = %8.3f %8.3f %8.3f"
            print(j_fmt % (j2angle*DPR, j2theta*DPR, j2omega) )
            print(b_fmt % \
                (b2LVel[0],b2LVel[1],b2LVel[2],b2AVel[0],b2AVel[1],b2AVel[2]) )
        linVel = vecCrossP(b1AVel,vecSub(j2Pos,j1Pos))
        Velode = vecAdd(vecCrossP(b2AVel,vecSub(b2Pos,j2Pos)),linVel)
        Velrk4 = m2Vel
        if PRINT_DATA:
            printVec('b2','Velode',Velode)  # Velode should be same as b2LVel
            printVec('b2','Velrk4',Velrk4)
            printVec('b2','b2AVel',b2AVel)
            printVec('b2','angVel2',angVel2)
        if PLOT_DATA:
            # for figures 3 and 5
            B2LVode[i] = vecMag(b2LVel)
            B2LVrk4[i] = vecMag(Velrk4)
            B2AVode[i] = b2AVel[2]*DPR  # b2AVel[2] should be same as j1omega+j2omega 
            B2AVrk4[i] = angVel2[2]*DPR
        # ... joint angular velocities from ODE and RK4
        if PLOT_DATA:
            # for figure 7
            J1AVode[i] = j1omega
            J2AVode[i] = j2omega
            J1AVrk4[i] = S[3]
            J2AVrk4[i] = S[4]
        # ... joint angular accelerations from ODE and RK4 
        Q      = (j1angle, j2angle)
        dQ     = (j1omega, j2omega)
        ddQode = (qdd_ode[0], qdd_ode[1])
        Q      = (S[1], S[2])
        dQ     = (S[3], S[4])
        ddQrk4 = calcAngAccel(Tq_rk4,Q,dQ)
        if PRINT_DATA:
            print("j1:  angle  = %8.4f  S[1]   = %8.4f" % (j1angle*DPR, S[1]*DPR) )
            print("j1:  omega  = %8.4f  S[3]   = %8.4f" % (j1omega, S[3]) )
            print("j1:  ddQode = %8.4f  ddQrk4 = %8.4f" % (ddQode[0],ddQrk4[0]) )
            print("j2:  angle  = %8.4f  S[2]   = %8.4f" % (j2angle*DPR, S[2]*DPR) )
            print("j2:  omega  = %8.4f  S[4]   = %8.4f" % (j2omega, S[4]) )
            print("j2:  ddQode = %8.4f  ddQrk4 = %8.4f" % (ddQode[1],ddQrk4[1]) )
        if PLOT_DATA:
            # for figure 8
            J1AAode[i] = ddQode[0]
            J2AAode[i] = ddQode[1]
            J1AArk4[i] = ddQrk4[0]
            J2AArk4[i] = ddQrk4[1]
        # ... joint torques from ODE and RK4
        if PRINT_DATA:
            print("j1:  Tq_ode = %8.4f  Tq_rk4 = %8.4f" % (Tq_ode[0],Tq_rk4[0]) )
            print("j2:  Tq_ode = %8.4f  Tq_rk4 = %8.4f" % (Tq_ode[1],Tq_rk4[1]) )
        if PLOT_DATA:
            # for figure 9
            J1TQode[i] = Tq_ode[0]  
            J2TQode[i] = Tq_ode[1]
            J1TQrk4[i] = Tq_rk4[0]
            J2TQrk4[i] = Tq_rk4[1]
        
        # Draw line for desired end-effector path
        pygame.draw.line(screen, PATH_COLOR, coord(path0), coord(path1), PATH_WIDTH)
        
        # Draw lines and circles representing ODE boxes, joints and body CoMs.
        pygame.draw.line(screen, BOX_COLOR, coord(j1Pos), coord(j2Pos), BOX_WIDTH)
        pygame.draw.line(screen, BOX_COLOR, coord(j2Pos), coord(j3Pos), BOX_WIDTH)
        pygame.draw.circle(screen, JNT_COLOR, coord(j1Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, JNT_COLOR, coord(j2Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, JNT_COLOR, coord(j3Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, BDY_COLOR, coord(b1Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, BDY_COLOR, coord(b2Pos, integer=True), cir_rad, 0)
        
        # Draw lines and circles representing robotic arm rods, pins and rod CoMs.
        pygame.draw.line(screen, ROD_COLOR, coord(p1Pos), coord(p2Pos), ROD_WIDTH)
        pygame.draw.line(screen, ROD_COLOR, coord(p2Pos), coord(p3Pos), ROD_WIDTH)
        pygame.draw.circle(screen, PIN_COLOR, coord(p1Pos, integer=True), cir_rad, 1)
        pygame.draw.circle(screen, PIN_COLOR, coord(p2Pos, integer=True), cir_rad, 1)
        pygame.draw.circle(screen, PIN_COLOR, coord(p3Pos, integer=True), cir_rad, 1)
        pygame.draw.circle(screen, COM_COLOR, coord(m1Pos, integer=True), cir_rad, 1)
        pygame.draw.circle(screen, COM_COLOR, coord(m2Pos, integer=True), cir_rad, 1)
        
        # Display updated screen.
        pygame.display.flip()
        if SAVE_ANIM:
            istr = format_string("%04d", i)
            fpth = "./anim/tutorial2arm_" + istr + ".png"
            pygame.image.save(screen, fpth)
            
        # Execute ODE simulation and RK4 integration.
        for n in range(N_STEP):
            # solve for torque
            tau = calcTorque(nSvar,S)
            # set ODE joint motors angular rate or torque
            if MOTOR_AVEL: am1.setParam(ode.ParamVel,S[3])
            else:          am1.addTorques(tau[0],0.0,0.0)
            if MOTOR_AVEL: am2.setParam(ode.ParamVel,S[4])
            else:          am2.addTorques(tau[1],0.0,0.0)
            # next ODE simulation step
            world.step(T_STEP)
            # next RK4 integration step
            S = rk4.step(S,dotS)

        # Update simulation time and data samples index.
        t = S[0]
        i = i + 1
        
        # Try to keep the specified framerate. 
        clk.tick(FPS)

    # Exited simulation loop.
    
    if PLOT_DATA:
        
        # Ensure Time data array contains nSamples of simulation time steps
        # in order to prevent plotted lines from wrapping back to time 0.0
        # if the simulation loop was terminated before all nSamples of data
        # were collected.
        
        while i < nSamples:
            Time[i] = t
            t = t + N_TIME
            i = i + 1
    
        # Create and show the plots.
        
        xlims = [0.0,T_STOP]               # x axis data limits
        l_col = ['r','r','g','g','b','b']  # line colors for plotted data sets
        l_typ = ['-',':','-',':','-',':']  # line styles for plotted data sets
        l_wid = [1.0,2.0,1.0,2.0,1.0,2.0]  # line widths for plotted data sets
        
        # Procedure to place figure in desktop window.
        
        def move_fig(fig):
            """
            Moves given figure plot window based on figure's number.
            """
            fign = fig.number
            x, y = 80*(fign+1), 40*(fign+1)
            backend = mpl.get_backend().upper()
            if backend[0:2] == 'WX':
                fig.canvas.manager.window.SetPosition((x,y))
            elif backend[0:2] == 'TK':
                fig.canvas.manager.window.wm_geometry("+%d+%d" % (x,y)) 
            else:  # QT or GTK
                fig.canvas.manager.window.move(x,y)
                
        # Procedure to create and show the plots
        
        def create_plot(fign, title, xlabel, ylabel, xlims, time,
                        l_dat, l_col, l_typ, l_wid, l_lbl):
            """
            Obtain plot figure number 'fign' and plot the 'l_dat' data sets
            versus time using the given axes and Line2D artist parameters.
            """
            fig = plt.figure(fign, figsize=(8,6), dpi=80)
            move_fig(fig)
            ax = fig.add_subplot(111, autoscale_on=False)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid()
            ax.set_xlim(xlims[0],xlims[1])
            # calculate and set figure axes y data limits
            ymin = min(l_dat[0]) 
            ymax = max(l_dat[0])
            ndat = len(l_dat)
            for i in range(ndat-1):
                ymin = min(ymin,min(l_dat[i+1]))
                ymax = max(ymax,max(l_dat[i+1]))
            ymin = ymin - 0.05*(ymax-ymin)
            ymax = ymax + 0.05*(ymax-ymin)
            ax.set_ylim(ymin,ymax,auto=True)
            # create Line2D artists for data sets and assign to figure axes
            lines = []
            for i in range(ndat):
                aline = Line2D(time,l_dat[i],c=l_col[i],
                               ls=l_typ[i],lw=l_wid[i],label=l_lbl[i],
                               marker=' ',mew=1.0,mec='k',mfc=l_col[i])
                ax.add_line(aline)
                lines.append(aline)
            return fig, ax, lines
        
        # Figures 1, 6, 7, 8 and 9 correspond to the five plots presented on
        # page 95 of reference [1], where figure 9 is for the Dynamics Results
        # Including Gravity case.
        
        (fig1, ax1, lines1) = create_plot(1,
            "Planar Robotic End-Effector Cartesian Pose (Px,Py,Theta)",
            "Time (sec)",
            "Px (r) and Py (g) (m), Theta (b) (rad)",
            xlims,
            Time,
            [EEXPode,EEXPrk4,EEYPode,EEYPrk4,EEARode,EEARrk4],
            l_col,l_typ,l_wid,
            ['Px (ODE)','Px (RK4)','Py (ODE)','Py (RK4)','Theta (ODE)', 'Theta (RK4)'])           
        if tStop < 0.6: legend_loc = 'center left'
        else:           legend_loc = 'upper right'          
        ax1.legend(loc=legend_loc)
        
        (fig2, ax2, lines2) = create_plot(2,
            "Planar Robotic End-Effector Velocity Components (Vx,Vy)",
            "Time (sec)",
            "Velocity Vx (r) and Vy (g) (m/sec)",
            xlims,
            Time,
            [EEXVode,EEXVrk4,EEYVode,EEYVrk4],
            l_col,l_typ,l_wid,
            ["Vx (ODE)","Vx (RK4)","Vy (ODE)","Vy (RK4)"])
        if tStop < 0.8: legend_loc = 'center right'
        else:           legend_loc = 'center left'         
        ax2.legend(loc=legend_loc)
        
        (fig3, ax3, lines3) = create_plot(3,
            "Planar Robotic Arm Body Linear Velocity",
            "Time (sec)",
            "Body 1 (r) and 2 (g) Lin Velocity (m/sec)",
            xlims,
            Time,
            [B1LVode,B1LVrk4,B2LVode,B2LVrk4],
            l_col,l_typ,l_wid,
            ["Body 1 (ODE)","Body 1 (RK4)","Body 2 (ODE)","Body 2 (RK4)"])
        if tStop < 0.9: legend_loc = 'center right'
        else:           legend_loc = 'upper left' 
        ax3.legend(loc=legend_loc)
                    
        (fig4, ax4, lines4) = create_plot(4,
            "Planar Robotic Arm Body Linear Acceleration",
            "Time (sec)",
            "Body 1 (r) and 2 (g) Lin Accel (m/sec/sec)'",
            xlims,
            Time,
            [B1LAode,B1LArk4,B1LAeom,B2LAode,B2LArk4,B2LAeom],
            ['r','r','r','g','g','g'],
            ['-',':','--','-',':','--'],
            [1.0,2.0,2.0,1.0,2.0,2.0],
            ["Body 1 (ODE)","Body 1 (RK4)","Body 1 (EOM)",
             "Body 2 (ODE)","Body 2 (RK4)","Body 2 (EOM)"])
        if tStop < 0.9: legend_loc = 'center right'
        else:           legend_loc = 'upper left' 
        ax4.legend(loc=legend_loc)
                    
        (fig5, ax5, lines5) = create_plot(5,
            "Planar Robotic Arm Body Angular Velocity",
            "Time (sec)",
            "Body 1 (r) and 2 (g) Ang Velocity (deg/sec)",
            xlims,
            Time,
            [B1AVode,B1AVrk4,B2AVode,B2AVrk4],
            l_col,l_typ,l_wid,
            ["Body 1 (ODE)","Body 1 (RK4)","Body 2 (ODE)","Body 2 (RK4)"])
        if tStop < 0.8: legend_loc = 'center right'
        else:           legend_loc = 'lower left'         
        ax5.legend(loc=legend_loc)
                    
        (fig6, ax6, lines6) = create_plot(6,
            "Planar Robotic Arm Joint Angles",
            "Time (sec)",
            "Joint 1 (r) and 2 (g) Angle (deg)",
            xlims,
            Time,
            [J1ADode,J1ADrk4,J2ADode,J2ADrk4],
            l_col,l_typ,l_wid,
            ["Joint 1 (ODE)","Joint 1 (RK4)","Joint 2 (ODE)","Joint 2 (RK4)"])
        if tStop < 0.6: legend_loc = 'center right'
        else:           legend_loc = 'center left'          
        ax6.legend(loc=legend_loc)
    
        (fig7, ax7, lines7) = create_plot(7,
            "Planar Robotic Arm Joint Rates",
            "Time (sec)",
            "Joint 1 (r) and 2 (g) Angle Rate (rad/sec)",
            xlims,
            Time,
            [J1AVode,J1AVrk4,J2AVode,J2AVrk4],
            l_col,l_typ,l_wid,
            ["Joint 1 (ODE)","Joint 1 (RK4)","Joint 2 (ODE)","Joint 2 (RK4)"])
        if tStop < 0.8: legend_loc = 'center right'
        else:           legend_loc = 'lower left'
        ax7.legend(loc=legend_loc)
                    
        (fig8, ax8, lines8) = create_plot(8,
            "Planar Robotic Arm Joint Accelerations",
            "Time (sec)",
            "Joint 1 (r) and 2 (g) Angle Accel (rad/sec^2)",
            xlims,
            Time,
            [J1AAode,J1AArk4,J2AAode,J2AArk4],
            l_col,l_typ,l_wid,
            ["Joint 1 (ODE)","Joint 1 (RK4)","Joint 2 (ODE)","Joint 2 (RK4)"])
        if tStop < 0.6: legend_loc = 'upper right'
        else:           legend_loc = 'lower left'       
        ax8.legend(loc=legend_loc)
                    
        (fig9, ax9, lines9) = create_plot(9,
            "Planar Robotic Arm Joint Torques",
            "Time (sec)",
            "Joint 1 (r) and 2 (g) Torque (Nm)",
            xlims,
            Time,
            [J1TQode,J1TQrk4,J2TQode,J2TQrk4],
            l_col,l_typ,l_wid,
            ["Joint 1 (ODE)","Joint 1 (RK4)","Joint 2 (ODE)","Joint 2 (RK4)"])
        if tStop < 0.8: legend_loc = 'center right'
        else:           legend_loc = 'center left' 
        ax9.legend(loc=legend_loc)
        
        # Block to keep plots displayed when not running interactively.
        plt.show(block=True)
        plt.close('all')
   
    # Wait till user closes pygame window to exit program...            
    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit()
                done = True
                