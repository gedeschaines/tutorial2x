#!/usr/bin/env ipython --matplotlib=qt

# pylint: disable=trailing-whitespace,bad-whitespace,invalid-name,anomalous-backslash-in-string

# File: tutorial2rods.py
# Auth: Gary E. Deschaines
# Date: 28 May 2015
# Prog: Double pendulum system modeled with PyODE, animated with Pygame
# Desc: Models numerical solution for double rod pendulum system dynamics 
#       originally presented as example 2 in PyODE tutorials.
#
#         http://pyode.sourceforge.net/tutorials/tutorial2.html
#
# PyODE example 2: Connecting bodies with joints
#
# modified by Gideon Klompje (removed literals and using
# 'ode.Mass.setSphereTotal' instead of 'ode.Mass.setSphere')
#
# modified by Gary E. Deschaines (changed double pendulum
# joints to hinge and bodies to rods, added integration 
# of the differential equations of motion using Runge-Kutta
# 4th order method (RK4), and utilized matplotlib to create
# plots of data collected from the ODE simulation and RK4 
# integration)
#
# References:
#
# [1] "Double pendulum," Wikipedia.org, obtained 13 May 2015 from: 
#     http://en.wikipedia.org/wiki/Double_pendulum, last modified
#     15 Jan 2015 at 02:24.
#
# [2] J. Awrejcewicz, Classical Mechanics: Dynamics, (Springer, 
#     New York, 2012)
#
# Disclaimer:
#
# See DISCLAIMER

import sys

from math import pi, ceil, floor, cos, sin, sqrt
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
    from vecMath import vecAdd, vecCrossP
    from vecMath import vecMag, vecMagSq, vecMulS, vecSub
    from vecMath import unitVec
    from vecMath import projectionVecAonUB, rejectionVecAfromUB
except ImportError:
    print("* Error: vecMath package required.")
    sys.exit()
  
try:
    import numpy             as np
    import matplotlib        as mpl
    import matplotlib.pyplot as plt
except ImportError:
    print("* Error: NumPy, SciPy and matplotlib packages required.")
    print("         Suggest installing the SciPy stack.")
    sys.exit()
  
try:
    from RK4_Solver import RK4_Solver
except ImportError:
    print("* Error: RK4_Solver class required.")
    sys.exit()
  
# Processing and output control flags
  
USE_PDOT = True   # Use differential equations of motion involving Pdot

T_STEP =  0.0005  # Simulation and integration time step size (sec)
T_STOP = 10.0     # Simulation and integration stop time (sec)

PRINT_DATA = False  # Controls printing of collected data
PRINT_FBCK = False  # Controls printing of joint feedback
PLOT_DATA  = True   # Controls plotting of collected data
SAVE_ANIM  = False  # Controls saving animation images

# Drawing constants

WINDOW_RESOLUTION = (640, 480)

DRAW_SCALE = WINDOW_RESOLUTION[0] / 5
"""Factor to multiply physical coordinates by to obtain screen size in pixels"""
DRAW_OFFSET = (WINDOW_RESOLUTION[0] / 2, 50) 
"""Screen coordinates (in pixels) that map to the physical origin (0, 0, 0)"""

FILL_COLOR = (255, 255, 255)  # background fill color
TEXT_COLOR = ( 10,  10,  10)  # text drawing color
CYL_WIDTH  = 8  # width of the line (in pixels) representing a cylinder
CYL_COLOR  = (  0,  50, 200)  # for drawing cylinder orientations from ODE
BDY_COLOR  = (  0,  50, 200)  # for drawing body positions from ODE
JNT_COLOR  = ( 50,  50,  50)  # for drawing joint positions from ODE
ROD_WIDTH  = 4  # width of the line (in pixels) representing a rod
ROD_COLOR  = (255,   0, 255)  # for drawing pendulum rod orientations from RK4
COM_COLOR  = (225,   0, 255)  # for drawing pendulum CoM positions from RK4
PIN_COLOR  = (255,   0,   0)  # for drawing pendulum pin positions from RK4

# Double Compound Pendulum System - characterization and initial conditions

"""
Pictogram of the double compound pendulum system.


              origin (anchor for joint 1)
                |
                | /\
                V/\ \
    ------------o--\-\-------> +x axis
                |\  \ \
                | \  \ \--{height of rods   
                |  \  \/
                |   \ /\
                |    + <--{1st rod center of mass (CoM)
                |     \  \
                |      \  \--{length of rods
                |=(+T1)>\  \
                |   ^    \ /
                V   :     O <--{tip of rod 1 (anchor for joint 2)
            -y axis :     |\
                    :     | \
                    :     |  \
       rod 1 angle}-:     |   \
                          |    + <--{2nd rod center of mass (CoM)
                          |     \ 
                          |      \                      
                          |=(+T2)>\ 
                          |   ^    \ 
                              :     O <--{tip of rod 2
                 rod 2 angle}-:


    Notes: (1) ODE body 1 corresponding to rod 1 is positioned wrt to the 
               origin and the initial angle of rod 1 is set to Theta0_1, 
               but the angle of ODE joint 1, which attaches body 1 to the 
               origin, is always initialized to zero by ODE. Consequently, 
               Theta0_1 must be added to the values returned by the ODE 
               getAngle method for joint 1 in order to match angle T1 
               depicted above.
             
           (2) ODE body 2 corresponding to rod 2 is positioned wrt to body
               1 and the initial angle of rod 2 is set to Theta0_2, but the
               angle of ODE joint 2, which which attaches body 2 to body 1, 
               is always initialized to zero by ODE. Since the angle value
               returned by the ODE getAngle method for joint 2 is equivalent
               to T2 minus T1 as depicted above, the sum of Theta0_2 and 
               the getAngle value for joint 1 must be added to the getAngle 
               value for joint 2 in order to match angle T2.
"""

ORIGIN  = (0.0, 0.0, 0.0)    # origin of inertial reference frame
Z_AXIS  = (0.0, 0.0, 1.0)    # axis of joint and body rotations
GRAVITY = (0.0, -9.81, 0.0)  # gravitational acceleration (m/sec/sec)

# Physical specifications

ROD_LENGTH = 1.0
ROD_HEIGHT = ROD_LENGTH/2.0
ROD_MASS   = 1.0

BODY_L    = ROD_LENGTH
BODY_LSQ  = BODY_L**2
BODY_H    = ROD_HEIGHT
BODY_R    = 0.0001  # extremely slender rod
BODY_RSQ  = BODY_R**2
BODY_V    = BODY_L*pi*BODY_R**2
BODY_M    = ROD_MASS
BODY_DEN  = BODY_M/BODY_V
BODY_Izzc = (BODY_M/12)*(3*BODY_RSQ +   BODY_LSQ)  # moment of inertia about CoM
BODY_Izzj = (BODY_M/12)*(3*BODY_RSQ + 4*BODY_LSQ)  # moment of inertia about joint

# Initial conditions
    
Theta0_1 = 135.0*RPD  # initial angle of 1st rod attached to origin
Theta0_2 =  90.0*RPD  # initial angle of 2nd rod attached to 1st rod

# Physical configuration

def distAngleToXYZ(dist,ang):
    """ 
    Converts given distance and angle (in radians) to xyz coordinate.
    """
    xyz = (dist*sin(ang), -dist*cos(ang), 0.0)
    return xyz

JOINT1_ANCHOR = ORIGIN
JOINT1_AXIS   = Z_AXIS

JOINT2_ANCHOR = vecAdd(JOINT1_ANCHOR,distAngleToXYZ(ROD_LENGTH,Theta0_1))
JOINT2_AXIS   = Z_AXIS

ROD1_POS = vecAdd(JOINT1_ANCHOR,distAngleToXYZ(ROD_HEIGHT,Theta0_1))
ROD1_DEN = BODY_DEN
ROD1_RAD = BODY_R
ROD1_LEN = BODY_L
    
ROD2_POS = vecAdd(JOINT2_ANCHOR,distAngleToXYZ(ROD_HEIGHT,Theta0_2))
ROD2_DEN = BODY_DEN
ROD2_RAD = BODY_R
ROD2_LEN = BODY_L

#-----------------------------------------------------------------------------
# System of ordinary differential equations (ode) characterizing the motion
# of a double compound pendulum composed of two identical slender rods as 
# described in the Wikipedia "Double Pendulum" page listed as reference [1].
# Specifically, the Lagrangian (L) equation presented for expressing the 
# total energy of the pendulum system using the generalized coordinates
# theta 1 and theta 2 (T1 and T2 in the pictogram respectively)
#
#   Ttrn = m*l*l*(5*thdot1^2 + thdot2^2 + 4*thdot1*thdot2*cos(th1-th2))/8
#   Trot = Izzc*(thdot1^2 + thdot2^2)/2
#   V    = -g*m*l*(3*cos(th1) + cos(th2))/2
#
#   L    = Ttrn + Trot - V
#   L    =  m*l*l*[thdot2^2 + 4*thdot1^2 + 3*thdot1*thdot2*cos(th1-th2)]/6
#        +  m*g*l*[3*cos(th1) + cos(th2)]/2
#
# was used to derive the following two generalized momenta equations for 
# the rods as p = d(L)/dthdot.
#
#   p1 = (1.0/6.0)*m*l*l*(8*thdot1 + 3*cos(th1-th2)*thdot2)       (1)
#   p2 = (1.0/6.0)*m*l*l*(3*cos(th1-th2)*thdot1 + 2*thdot2)       (2)
#
# where
#
#   Ttrn   = translational kinetic energy
#   Trot   = rotational kinetic energy
#   V      = positional potential energy
#   m      = mass of pendulum rod
#   l      = length of pendulum rod
#   Izzc   = moment of inertia about the center of mass (approx. m*l*l/12)
#   g      = gravitational acceleration 
#   th#    = rod # angle (theta) wrt to -y axis (+ counter-clockwise)
#   p#     = momentum (p) of rod #
#   thdot# = d(th#)/dt
#   pdot#  = d(p#)/dt
#   delth  = th1 - th2
#
# Equations (1) and (2) can be arranged in matrix form such that [p1 p2]' 
# = [M]*[thdot1 thdot2]' (where [ ]' denotes matrix transpose) inorder to 
# solve for [thdot1 thdot2]' as inv(M)*[p1 p2]', 
#   
#   | p1 |   | (4/3)*m*l*l  (1/2)*m*l*l*cos(th1-th2) |   | thdot1 |
#   |    | = |                                       | * |        |
#   | p2 |   | (1/2)*m*l*l*cos(th1-th2)  (1/3)*m*l*l |   | thdot2 |
#
# yielding the following expressions for d(theta)/dt
#
#   thdot1 = 6*(2*p1 - 3*cos(delth)*p2)/(m*l*l*(16 - 9*cos(delth)^2)
#   thdot2 = 6*(8*p2 - 3*cos(delth)*p1)/(m*l*l*(16 - 9*cos(delth)^2)
#
# The derivatives d(L)/dtheta for theta 1 and theta 2 yield the following
# expressions for d(p)/dt. 
#
#   pdot1 = -0.5*m*l*(3*g*sin(th1) + l*thdot1*thdot2*sin(delth)) 
#   pdot2 = -0.5*m*l*(g*sin(th2) - l*thdot1*thdot2*sin(delth))
#
# The state variables (theta1, theta2, p1, p2) and associated equations 
# for their time derivatives (thdot1, thdot2, pdot1, pdot2) comprise 
# the "Wikipedia" model. For comparison, an alternate set of differential
# equations of motion involving theta, d(theta)/dt and d(d(theta)/dt)/dt
# derived from material presented as the "Equations of Motion" for "Planar
# Dynamics of a Triple Physical Pendulum" in section 2.3.1 on pages 83-90 
# of reference [2] comprise the "Springer" model.
#
#      Wikipedia Model                   Springer Model
#   ----------------------        ----------------------------  
#    dS[1] = d(theta1)/dt          dS[1] = d(theta1)/dt
#    dS[2] = d(theta2)/dt          dS[2] = d(theta2)/dt
#    dS[3] = d(p1)/dt              dS[3] = d(d(theta1)/dt)/dt
#    dS[4] = d(p2)/dt              dS[4] = d(d(theta2)/dt)/dt
#
# The dotS function incorporates the sets of first order differential 
# equations of motion for both models. A Runge-Kutta 4th order integration 
# method is applied to the dotS function to develop the motion of the 
# pendulum system in terms of the state variables S[] by solving for S[] 
# at each time step h from t0 to tStop using the following algorithm
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
#   [1] "Double pendulum," Wikipedia.org, obtained 13 May 2015 from: 
#       http://en.wikipedia.org/wiki/Double_pendulum, last modified
#       15 Jan 2015 at 02:24.
#
#   [2] J. Awrejcewicz, Classical Mechanics: Dynamics, (Springer, 
#       New York, 2012)
#

Theta1 = Theta0_1   # initial angle of 1st rod attached to origin
Theta2 = Theta0_2   # initial angle of 2nd rod attached to 1st rod
Tdot1  =  0.0*RPD   # initial angular velocity of first rod
Tdot2  =  0.0*RPD   # initial angular velocity of second rod
tStep  = T_STEP     # simulation and integration time step (sec)
tStop  = T_STOP     # simulation and integration stop time (sec)

# Differential equations of motion constants

g     = vecMag(GRAVITY)
gdivl = g/ROD_LENGTH
mlsq  = ROD_MASS*ROD_LENGTH**2
hmlsq = 0.5*mlsq
mgl   = ROD_MASS*g*ROD_LENGTH
r     = ROD_LENGTH
sixth = 1.0/6.0

# Initial potential energy of system

PE0 = -2*ROD_MASS*g*ROD_LENGTH +\
         ROD_MASS*g*ROD_HEIGHT*cos(Theta0_1) +\
         ROD_MASS*g*(ROD_LENGTH*cos(Theta0_1) + ROD_HEIGHT*cos(Theta0_2))
         
# Initial values for state variables array

nSvar = 5                # number of state variables
S     = np.zeros(nSvar)  # state variables
dS    = np.zeros(nSvar)  # state derivatives

if USE_PDOT:
    cosdth = cos(Theta1 - Theta2)
    p1     = sixth*mlsq*(8*Tdot1 + 3*Tdot2*cosdth)  # angular momentum p1
    p2     = sixth*mlsq*(2*Tdot2 + 3*Tdot1*cosdth)  # angular momentum p2
    S[0]   = 0.0      # t0 
    S[1]   = Theta1
    S[2]   = Theta2
    S[3]   = p1
    S[4]   = p2
else:
    S[0] = 0.0     # t0 
    S[1] = Theta1
    S[2] = Theta2
    S[3] = Tdot1   # d(theta1)/dt
    S[4] = Tdot2   # d(theta2)/dt

def dotS(n,S):
    """
    State derivatives function.
    """
    global gdivl, mlsq, hmlsq, r
  
    dS = np.zeros(n) 
    
    if USE_PDOT:
        th1 = S[1]  # angle theta1
        th2 = S[2]  # angle theta2
        p1  = S[3]  # angular momentum p1
        p2  = S[4]  # angular momentum p2
    
        dth      = th1 - th2
        cosdth   = cos(dth)
        cosdthsq = cos(dth)**2
        sindth   = sin(dth)
        numer1   = 6*(2*p1 - 3*cosdth*p2)
        numer2   = 6*(8*p2 - 3*cosdth*p1)
        denom    = mlsq*(16 - 9*cosdthsq)
    
        dS[0] = 1.0                                             # d(t)/dt
        dS[1] = numer1/denom                                    # d(th1)/dt
        dS[2] = numer2/denom                                    # d(th2)/dt
        dS[3] = -hmlsq*(dS[1]*dS[2]*sindth + 3*gdivl*sin(th1))  # d(p1)/dt
        dS[4] = -hmlsq*(gdivl*sin(th2) - dS[1]*dS[2]*sindth)    # d(p2)/dt
        
    else:
        th1    = S[1]  # theta1
        th2    = S[2]  # theta2
        thdot1 = S[3]  # d(theta1)/dt
        thdot2 = S[4]  # d(theta2)/dt
        
        thdot1sq = thdot1**2
        thdot2sq = thdot2**2
        dth      = th1 - th2
        sinth1   = sin(th1)
        sinth2   = sin(th2)
        cosdth   = cos(dth)
        cosdthsq = cos(dth)**2
        sindth   = sin(dth)
        sind21th = sin(2*th1-th2)
        sind22th = sin(2*th1-2*th2)
        numer1   = 3*(r*sindth*(3*cosdth*thdot1sq + 2*thdot2sq) + \
                        3*g*(2*sinth1 - cosdth*sinth2))
        numer2   = 3*(r*(16*sindth*thdot1sq + 3*sind22th*thdot2sq) + \
                        g*(9*sind21th - 7*sinth2))
        denom    = r*(9*cosdthsq - 16)
        
        dS[0] =  1.0               # d(t)/dt
        dS[1] =  thdot1            # d(th1)/dt
        dS[2] =  thdot2            # d(th2)/dt
        dS[3] =  numer1/denom      # d(thdot1)/dt
        dS[4] = -numer2/(2*denom)  # d(thdot2)/dt
        
    return dS
    
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
        
def vecRotZ(V, rot):
    """
    Performs 2D transformation on given XYZ vector 'V' to yield 
    an xyz vector in a coordinate frame rotated about the Z/z 
    axis by the given 'rot' angle in radians.
    """
    cosr = cos(rot)
    sinr = sin(rot)
      
    x =  cosr*V[0] + sinr*V[1] 
    y = -sinr*V[0] + cosr*V[1]
    z = V[2]
      
    return (x, y, z)
     
def printFeedback(name, t, fb):
    """
    Prints elements of a joint feedback structure.
    """
    print("fb:  %s, t=%8.3f" % (name, t))
    
    F1 = vecMag(fb[0])
    T1 = vecMag(fb[1])
    print("  F1:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
            (F1, fb[0][0], fb[0][1], fb[0][2]) )
    print("  T1:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
            (T1, fb[1][0], fb[1][1], fb[1][2]) )
            
    if name == 'j2':
        F2 = vecMag(fb[2])
        T2 = vecMag(fb[3])
        print("  F2:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
                (F2, fb[2][0], fb[2][1], fb[2][2]) )
        print("  T2:  %10.3f (%8.3f | %8.3f | %8.3f)" % \
                (T2, fb[3][0], fb[3][1], fb[3][2]) )
                
def printVec(body,name,V):
    """
    Print name, components and magnitude of given vector.
    """
    fmt = "%s:  %s = %9.4f %9.4f %9.4f  %10.4f"
    
    print(fmt % (body, name, V[0], V[1], V[2], vecMag(V)) )
    
#-----------------------------------------------------------------------------

# Initialize pygame
pygame.init()

# Open a display
screen = pygame.display.set_mode(WINDOW_RESOLUTION)

# Create a ODE world object
world = ode.World()
world.setGravity(GRAVITY)
world.setERP(1.0)
world.setCFM(0.0)

# Create two bodies
body1 = ode.Body(world)
mass1 = ode.Mass()
mass1.setCylinderTotal(ROD_MASS, 2, ROD1_RAD, ROD1_LEN)
body1.setMass(mass1)
body1.setPosition(ROD1_POS)
body1.setQuaternion((cos(Theta0_1/2),0,0,sin(Theta0_1/2)))

body2 = ode.Body(world)
mass2 = ode.Mass()
mass2.setCylinderTotal(ROD_MASS, 2, ROD2_RAD, ROD2_LEN)
body2.setMass(mass2)
body2.setPosition(ROD2_POS)
body2.setQuaternion((cos(Theta0_2/2),0,0,sin(Theta0_2/2)))

# Connect body1 with the static environment
j1 = ode.HingeJoint(world)
j1.attach(body1, ode.environment)
j1.setAnchor(JOINT1_ANCHOR)
j1.setAxis(JOINT1_AXIS)
j1.setFeedback(True)

# Connect body2 with body1
j2 = ode.HingeJoint(world)
j2.attach(body2, body1)
j2.setAnchor(JOINT2_ANCHOR)
j2.setAxis(JOINT2_AXIS)
j2.setFeedback(True)

# Define pygame circle radius for drawing each joint sphere
cir_rad = int(CYL_WIDTH)

# Create background for text and clearing screen
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(FILL_COLOR)

# Write title
if pygame.font:
    title   = "PyODE Tutorial 2 - Double Compound Pendulum Simulation"
    font    = pygame.font.Font(None, 30)
    text    = font.render(title, 1, TEXT_COLOR)
    textpos = text.get_rect(centerx=background.get_width()/2)
    background.blit(text, textpos)
    font = pygame.font.Font(None, 24)
    
# Clear screen
screen.blit(background, (0, 0))

# Simulation loop...

FPS    = 25
F_TIME = 1.0/FPS
N_STEP = int(floor((F_TIME + 0.5*T_STEP)/T_STEP))
N_TIME = N_STEP*T_STEP

if __name__ == "__main__":
    
    # Create simulation data collection arrays for plotting.
    
    nSamples = int(ceil(T_STOP/N_TIME)) + 1
    
    Time    = np.zeros(nSamples)  # simulation time
    B1LVode = np.zeros(nSamples)  # body 1 linear velocity from ODE
    B1AVode = np.zeros(nSamples)  # body 1 angular velocity from ODE
    B1LVrk4 = np.zeros(nSamples)  # body 1 linear velocity from RK4
    B1AVrk4 = np.zeros(nSamples)  # body 1 angular velocity from RK4
    B2LVode = np.zeros(nSamples)  # body 2 linear velocity from ODE
    B2AVode = np.zeros(nSamples)  # body 2 angular velocity from ODE
    B2LVrk4 = np.zeros(nSamples)  # body 2 linear velocity from RK4
    B2AVrk4 = np.zeros(nSamples)  # body 2 angular velocity from RK4
    TOTEode = np.zeros(nSamples)  # total system energy from ODE
    TOTErk4 = np.zeros(nSamples)  # total system energy from RK4

    # Instantiate clock to regulate display updates. 
    
    clk = pygame.time.Clock()
    
    # Set initial linear and angular velocities of the bodies based on the
    # initial angular velocities (thdot#) of the pendulum rods.
    
    j1Pos = j1.getAnchor()
    j2Pos = j2.getAnchor()
    b1    = j1.getBody(0)
    b2    = j2.getBody(0)
    b1Pos = b1.getPosition()
    b2Pos = b2.getPosition()
    
    if USE_PDOT:
        dS = dotS(nSvar,S)
        angVel1 = vecMulS(JOINT1_AXIS,dS[1])
        angVel2 = vecMulS(JOINT2_AXIS,dS[2])
    else:
        angVel1 = vecMulS(JOINT1_AXIS,S[3])
        angVel2 = vecMulS(JOINT2_AXIS,S[4])
        
    b1.setLinearVel(vecCrossP(angVel1,vecSub(b1Pos,j1Pos)))
    b1.setAngularVel(angVel1)
    linVel = vecCrossP(angVel1,vecSub(j2Pos,j1Pos))
    b2.setLinearVel(vecAdd(vecCrossP(angVel2,vecSub(b2Pos,j2Pos)),linVel))
    b2.setAngularVel(angVel2)
    
    # Instantiate Runge-Kutta 4th order ode solver and initialize.
    
    rk4 = RK4_Solver(tStep,nSvar)
    
    rk4.init(S)
    
    # Take a very small step in ODE to set rates and feedback data.
    
    world.step(T_STEP/1000.0)
    
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
            tformat   = "Simulation Time = %8.3f (sec)"
            text      = font.render(tformat % t, 1, TEXT_COLOR)
            textpos   = text.get_rect(centerx=background.get_width()/2)
            hoffset   = int(ceil(1.5*textpos.height))
            textpos.y = background.get_height() - hoffset
            screen.blit(text, textpos)
            
        # Get joint relative angles and angular rates.
        j1angle = j1.getAngle()
        j1omega = j1.getAngleRate()
        j2angle = j2.getAngle()
        j2omega = j2.getAngleRate()
        
        # Convert joint angles to absolute (refer to pictogram notes).
        j1theta = Theta0_1 + j1angle
        j2theta = Theta0_2 + j1angle + j2angle  
        
        # Get current ODE joint and body world space positions.
        j1Pos = j1.getAnchor()
        j2Pos = j2.getAnchor()
        j3Pos = vecAdd(distAngleToXYZ(ROD_LENGTH,j2theta),j2Pos)  # tip of rod 2
        b1    = j1.getBody(0)
        b2    = j2.getBody(0)
        b1Pos = b1.getPosition()
        b2Pos = b2.getPosition()
        
        # Get body linear and angular velocities.
        b1LVel = b1.getLinearVel()
        b1AVel = b1.getAngularVel()
        b2LVel = b2.getLinearVel()
        b2AVel = b2.getAngularVel()
        
        # Get current pendulum system angular rates.
        if USE_PDOT:
            dS = dotS(nSvar,S)
            angVel1 = vecMulS(JOINT1_AXIS,dS[1])
            angVel2 = vecMulS(JOINT2_AXIS,dS[2])
        else:
            angVel1 = vecMulS(JOINT1_AXIS,S[3])
            angVel2 = vecMulS(JOINT2_AXIS,S[4])
            
        # Get current double pendulum pin and mass world space positions.
        p1Pos = ORIGIN
        p2Pos = vecAdd(distAngleToXYZ(ROD_LENGTH,S[1]),p1Pos)
        p3Pos = vecAdd(distAngleToXYZ(ROD_LENGTH,S[2]),p2Pos)  # tip of rod 2
        m1Pos = vecAdd(distAngleToXYZ(ROD_HEIGHT,S[1]),p1Pos)
        m2Pos = vecAdd(distAngleToXYZ(ROD_HEIGHT,S[2]),p2Pos)
        
        # Get joint feedback forces and torques
        fb1 = j1.getFeedback()
        fb2 = j2.getFeedback()
        if PRINT_FBCK:
            print("t : %7.3f" % t)
            printFeedback('j1',t,fb1)
            printFeedback('j2',t,fb2)
        
        # Evaluate body dynamics
        if PRINT_FBCK and not USE_PDOT:
            
            print("b1:  theta = %8.3f  omega = %8.4f" % (j1theta*DPR, j1omega) )
            print("b2:  theta = %8.3f  omega = %8.4f" % (j2theta*DPR, j2omega) )
            
            b1Fmom = vecCrossP(vecSub(b1Pos,j2Pos),fb2[2])
            b1Ttot = vecAdd(fb1[1],fb2[3])
            b1Mtot = vecAdd(b1Ttot,b1Fmom)
            alpha1 = vecMag(b1Mtot)/BODY_Izzc
                        
            b2Fmom = vecCrossP(vecSub(b2Pos,j2Pos),fb2[0])
            b2Ttot = fb2[1]
            b2Mtot = vecAdd(b2Ttot,b2Fmom)
            alpha2 = vecMag(fb2[1])/BODY_Izzc
            
            fmt = "%s:  Fmom = %8.3f  Ttot = %8.3f  Mtot = %8.3f  alpha = %8.3f"
            print(fmt % ('b1',vecMag(b1Fmom), vecMag(b1Ttot), vecMag(b1Mtot), alpha1) )
            print(fmt % ('b2',vecMag(b2Fmom), vecMag(b2Ttot), vecMag(b2Mtot), alpha2) )
            
            # ... body Com acceleration from rotational moments
            angVelJ1 = vecMulS(JOINT1_AXIS,j1omega)
            linVelB1 = vecCrossP(angVelJ1,vecSub(b1Pos,j1Pos))
            linVelJ2 = vecCrossP(angVelJ1,vecSub(j2Pos,j1Pos))
            angVelJ2 = vecMulS(JOINT2_AXIS,j1omega+j2omega)
            linVelB2 = vecAdd(vecCrossP(angVelJ2,vecSub(b2Pos,j2Pos)),linVelJ2)
            
            printVec('j1', 'angVel', angVelJ1)
            printVec('j2', 'angVel', angVelJ2)
            printVec('b1', 'angVerr', vecSub(angVelJ1,b1AVel) )
            printVec('b2', 'angVerr', vecSub(angVelJ2,b2AVel) )
            
            printVec('b1', 'linVel', linVelB1)
            printVec('j2', 'linVel', linVelJ2)
            printVec('b2', 'linVel', linVelB2)
            printVec('b1', 'linVerr', vecSub(linVelB1,b1LVel) )
            printVec('b2', 'linVerr', vecSub(linVelB2,b2LVel) )
            
            # ... unit direction vector from body 2 to joint 2
            Uvec2  = unitVec(vecSub(j2Pos,b2Pos))
            
            # ... rotational velocities at joint 2 and body 2
            VwJ2n = projectionVecAonUB(linVelJ2,Uvec2)
            VwJ2t = rejectionVecAfromUB(linVelJ2,Uvec2)
            VwB2n = projectionVecAonUB(linVelB2,Uvec2)
            VwB2t = rejectionVecAfromUB(linVelB2,Uvec2)
            
            # ... rotational accelerations at joint 2 and body 2
            AwJ2  = vecCrossP(angVelJ1,vecCrossP(angVelJ1,vecSub(j2Pos,j1Pos)))
            AwJ2n = projectionVecAonUB(AwJ2,Uvec2)
            AwJ2t = rejectionVecAfromUB(AwJ2,Uvec2)
            AwB2  = vecCrossP(angVelJ2,vecCrossP(angVelJ2,vecSub(b2Pos,j2Pos)))
            AwB2n = projectionVecAonUB(AwB2,Uvec2)
            AwB2t = rejectionVecAfromUB(AwB2,Uvec2)
            
            # ... tangential forces at joint 2 and body 2
            FgB2t = vecMulS(rejectionVecAfromUB(GRAVITY,Uvec2),ROD_MASS)
            FwB2t = vecMulS(AwB2t,ROD_MASS)
            FwJ2t = vecMulS(AwJ2t,ROD_MASS)
            FrJ2t = rejectionVecAfromUB(fb2[0],Uvec2)
            
            # ... normal forces at joint 2 and body 2
            FgB2n = vecMulS(projectionVecAonUB(GRAVITY,Uvec2),ROD_MASS)
            FwB2n = vecMulS(AwB2n,ROD_MASS)
            FwJ2n = vecMulS(AwJ2n,ROD_MASS)
            FrJ2n = projectionVecAonUB(fb2[0],Uvec2)
            
            print("Vectors in body 2 frame (tangential & normal axes")
            printVec('j2', 'fb2[0]', vecRotZ(fb2[0],j2theta))
            printVec('j2', 'fb2[1]', vecRotZ(fb2[1],j2theta))
            printVec('J2', 'linVel', vecRotZ(linVelJ2,j2theta))
            printVec('b2', 'linVel', vecRotZ(linVelB2,j2theta))
            
            print("Vectors in body 2 frame - tangential axis")
            printVec('j2', 'VwJ2t', vecRotZ(VwJ2t,j2theta))
            printVec('b2', 'VwB2t', vecRotZ(VwB2t,j2theta))
            printVec('j2', 'AwJ2t', vecRotZ(AwJ2t,j2theta))
            printVec('b2', 'AwB2t', vecRotZ(AwB2t,j2theta))
            printVec('b2', 'FgB2t', vecRotZ(FgB2t,j2theta))
            printVec('b2', 'FwB2t', vecRotZ(FwB2t,j2theta))
            printVec('j2', 'FwJ2t', vecRotZ(FwJ2t,j2theta))
            printVec('j2', 'FrJ2t', vecRotZ(FrJ2t,j2theta))
            
            print("Vectors in body 2 frame - normal axis")
            printVec('j2', 'VwJ2n', vecRotZ(VwJ2n,j2theta))
            printVec('b2', 'VwB2n', vecRotZ(VwB2n,j2theta))
            printVec('j2', 'AwJ2n', vecRotZ(AwJ2n,j2theta))
            printVec('b2', 'AwB2n', vecRotZ(AwB2n,j2theta))
            printVec('b2', 'FgB2n', vecRotZ(FgB2n,j2theta))
            printVec('b2', 'FwB2n', vecRotZ(FwB2n,j2theta))
            printVec('j2', 'FwJ2n', vecRotZ(FwJ2n,j2theta))
            printVec('j2', 'FrJ2n', vecRotZ(FrJ2n,j2theta))
            
            dS = dotS(nSvar,S)
            
            b1Acct = dS[3]*ROD_HEIGHT 
            b1Accn = S[3]**2*ROD_HEIGHT 
            b1Acc  = sqrt(b1Acct**2 + b1Accn**2)
            
            j2Acct = dS[3]*ROD_LENGTH 
            j2Accn = S[3]**2*ROD_LENGTH 
            j2Acc  = sqrt(j2Acct**2 + j2Accn**2)
            
            b2Acct = dS[4]*ROD_HEIGHT + \
                ROD_LENGTH*(dS[3]*cos(S[2]-S[1]) - S[3]**2*sin(S[2]-S[1]))
            b2Accn = S[4]**2*ROD_HEIGHT - \
                ROD_LENGTH*(dS[3]*sin(S[2]-S[1]) - S[3]**2*cos(S[2]-S[1]))
            b2Acc  = sqrt(b2Acct**2 + b2Accn**2)
            
            j3Acct = dS[4]*ROD_LENGTH + \
                ROD_LENGTH*(dS[3]*cos(S[2]-S[1]) + S[3]**2*sin(S[2]-S[1]))
            j3Accn = S[4]**2*ROD_LENGTH - \
                ROD_LENGTH*(dS[3]*sin(S[2]-S[1]) - S[3]**2*cos(S[2]-S[1]))
            j3Acc  = sqrt(j3Acct**2 + j3Accn**2)
            
            fmt = "%s:  Tddot = %8.4f  Cacc = %8.3f  Oacc = %8.3f"
            print(fmt % ('b1',dS[3],b1Acc,j2Acc) )
            print(fmt % ('b2',dS[4],b2Acc,j3Acc) )
            
        # Collect data for plotting.
        # ... time steps
        if PRINT_DATA: print("t : %7.3f" % t)
        Time[i] = t
        # ... linear and angular velocity for body 1 from ODE and RK4
        if PRINT_DATA:
            j_fmt = "j1:  theta = %8.3f  omega = %8.3f"
            b_fmt = "b1:  LVel = %8.3f %8.3f %8.3f  AVel = %8.3f %8.3f %8.3f"
            print(j_fmt % (j1theta*DPR, j1omega*DPR) )
            print(b_fmt % \
                (b1LVel[0],b1LVel[1],b1LVel[2],b1AVel[0],b1AVel[1],b1AVel[2]) )
        Velode = vecCrossP(b1AVel,vecSub(b1Pos,j1Pos)) 
        Velrk4 = vecCrossP(angVel1,vecSub(m1Pos,p1Pos))
        if PRINT_DATA:
            fmt = "b1:  %s = %8.3f %8.3f %8.3f"
            print(fmt % ('Velode',Velode[0],Velode[1],Velode[2]) )
            print(fmt % ('Velrk4',Velrk4[0],Velrk4[1],Velrk4[2]) )
        B1LVode[i] = vecMag(Velode)  # Velode should be same as b1LVel
        B1AVode[i] = j1omega         # j1omega should be same as b1AVel[2] 
        B1LVrk4[i] = vecMag(Velrk4)
        if USE_PDOT: B1AVrk4[i] = dS[1]
        else       : B1AVrk4[i] = S[3]
        # ... linear and angular velocity for body 2 from ODE and RK4
        if PRINT_DATA:
            j_fmt = "j2:  theta = %8.3f  omega = %8.3f"
            b_fmt = "b2:  LVel = %8.3f %8.3f %8.3f  AVel = %8.3f %8.3f %8.3f"
            print(j_fmt % (j2theta*DPR, j2omega*DPR) )
            print(b_fmt % \
                (b2LVel[0],b2LVel[1],b2LVel[2],b2AVel[0],b2AVel[1],b2AVel[2]) )
        linVel = vecCrossP(b1AVel,vecSub(j2Pos,j1Pos))
        Velode = vecAdd(vecCrossP(b2AVel,vecSub(b2Pos,j2Pos)),linVel)
        linVel = vecCrossP(angVel1,vecSub(p2Pos,p1Pos))
        Velrk4 = vecAdd(vecCrossP(angVel2,vecSub(m2Pos,p2Pos)),linVel)
        if PRINT_DATA:
            fmt = "b2:  %s = %8.3f %8.3f %8.3f"
            print(fmt % ('Velode',Velode[0],Velode[1],Velode[2]) )
            print(fmt % ('Velrk4',Velrk4[0],Velrk4[1],Velrk4[2]) )
        B2LVode[i] = vecMag(Velode)     # Velode should be same as b2LVel
        B2AVode[i] = (j1omega+j2omega)  # j1omega+j2omega should be same as b2AVel[2]
        B2LVrk4[i] = vecMag(Velrk4)
        if USE_PDOT: B2AVrk4[i] = dS[2]
        else       : B2AVrk4[i] = S[4]
        # ... translational kinetic energy for ODE bodies
        b1KEtrn = 0.0  # CoM of body 1 only rotates about joint 1
        b2KEtrn = 0.5*ROD_MASS*vecMagSq(b2LVel)
        # ... rotational kinetic energy for ODE bodies
        b1KErot = 0.5*BODY_Izzj*vecMagSq(b1AVel)
        b2KErot = 0.5*BODY_Izzc*vecMagSq(b2AVel)
        # ... potential energy for ODE bodies (relative to Theta1=0 & Theta2=0)
        b1PE = ROD_MASS*g*(b1Pos[1] + ROD_HEIGHT)
        b2PE = ROD_MASS*g*(b2Pos[1] + ROD_LENGTH + ROD_HEIGHT)
        # ... total system energy for ODE bodies
        TOTEode[i] = b1KEtrn + b2KEtrn + b1KErot + b2KErot + b1PE + b2PE + PE0
        # ... kinetic, potential and total energy for RK4 rods
        if USE_PDOT:
            kinEtrn = mlsq*(5*dS[1]**2 + dS[2]**2 + 4*dS[1]*dS[2]*cos(S[1]-S[2]))/8
            kinErot = BODY_Izzc*(dS[1]**2 + dS[2]**2)/2
            potE    = 2*mgl - mgl*(3*cos(S[1]) + cos(S[2]))/2
        else:
            kinEtrn = mlsq*(5*S[3]**2 + S[4]**2 + 4*S[3]*S[4]*cos(S[1]-S[2]))/8          
            kinErot = BODY_Izzc*(S[3]**2 + S[4]**2)/2
            potE    = 2*mgl - mgl*(3*cos(S[1]) + cos(S[2]))/2
        TOTErk4[i] = kinEtrn + kinErot + potE + PE0
        if PRINT_DATA:
            fmt = "  :  totEode = %10.4f  totErk4 = %10.4f"
            print(fmt % (TOTEode[i], TOTErk4[i]) )
        
        # Draw lines and circles representing ODE cylinders, joints and body CoMs.
        pygame.draw.line(screen, CYL_COLOR, coord(j1Pos), coord(j2Pos), CYL_WIDTH)
        pygame.draw.line(screen, CYL_COLOR, coord(j2Pos), coord(j3Pos), CYL_WIDTH)
        pygame.draw.circle(screen, JNT_COLOR, coord(j1Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, JNT_COLOR, coord(j2Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, JNT_COLOR, coord(j3Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, BDY_COLOR, coord(b1Pos, integer=True), cir_rad, 0)
        pygame.draw.circle(screen, BDY_COLOR, coord(b2Pos, integer=True), cir_rad, 0)
        
        # Draw lines and circles representing pendulum rods, pins and rod CoMs.
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
            fpth = "./anim/tutorial2rods_" + istr + ".png"
            pygame.image.save(screen, fpth)
            
        # Execute ODE simulation and RK4 integration
        for n in range(N_STEP):
            # Next ODE simulation step.
            world.step(T_STEP)
            # Next RK4 integration step.
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
    
        figures = []
            
        figures.append(plt.figure(1, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 1 ||Lin Vel||")
        plt.xlabel('Time (sec)')
        plt.ylabel('Absolute Linear Velocity (m/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B1LVode,'b-',Time,B1LVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='lower left')
    
        figures.append(plt.figure(2, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 1 Angular Velocity")
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Velocity (rad/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B1AVode,'b-',Time,B1AVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='lower left')
    
        figures.append(plt.figure(3, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 2 ||Lin Vel||")
        plt.xlabel('Time (sec)')
        plt.ylabel('Absolute Linear Velocity (m/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B2LVode,'b-',Time,B2LVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='lower left')
    
        figures.append(plt.figure(4, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 2 Angular Velocity")
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Velocity (rad/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B2AVode,'b-',Time,B2AVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='lower left')
        
        figures.append(plt.figure(5, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum System Total Energy")
        plt.xlabel('Time (sec)')
        plt.ylabel('Total Energy (J)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,TOTEode,'b-',Time,TOTErk4,'r:2')
        plt.legend(('ODE','RK4'),loc='lower left')
        
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
            else:  # Qt or GTK
                fig.canvas.manager.window.move(x,y)
            
        for fig in figures:
            move_fig(fig)
            
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
                