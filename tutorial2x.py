#!/usr/bin/env ipython --matplotlib=qt

# pylint: disable=trailing-whitespace,bad-whitespace,invalid-name,anomalous-backslash-in-string

# File: tutorial2x.py
# Auth: Gary E. Deschaines
# Date: 18 May 2015
# Prog: Double pendulum system modeled with PyODE, animated with Pygame
# Desc: Models numerical solution for double pendulum system dynamics 
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
# joints to hinge, restricted pendulum to equal length rods
# and equal mass bodies, added integration of linearized or
# non-linearized differential equations of motion (eqom) using
# Runge-Kutta 4th order method (RK4), and utilized matplotlib
# to generate plots of data collected from the ODE simulation
# and RK4 integration)
#
# References (as indexed in other tutorial2x Python scripts):
#
# [1] Greenwood, Donald T., "Principles of Dynamics." Prentice-Hall,
#     Inc.: Englewood Cliffs, N.J., 1965.
#
# [2] Nielsen, R.O., Have E., Nielsen, B.T. "The Double Pendulum: First Year
#     Project." The Niels Bohr Institute, Mar. 21, 2013. Web available at
#     http://psi.nbi.dk/@psi/wiki/The%20Double%20Pendulum/files/projekt%202013-14%20RON%20EH%20BTN.pdf
#
# [5] Lynch, Kevin M. and Park, Frank C., "Modern Robotics:  Mechanics, 
#     Planning, and Control," 3rd printing 2019, Cambridge University
#     Press, 2017. Web available at
#     http://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf
#
# [6] Craig, John J., "Introduction to Robotics: Mechanics and Control," 3rd
#     ed., Pearson/Prentice-Hall, Upper Saddle River, N.J., 2005.
#
# [7] Liu, Karen and Jain, Sumit, "A Quick Tutorial on Multibody Dynamics,"
#     Tech Report GIT-GVU-15-01-1, School of Interactive Computing, Georgia
#     Institute of Technology, 2012. Web available at
#     https://bitbucket.org/karenliu/rtql8/src/default/docs/dynamics-tutorial.pdf
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
    from vecMath import vecAdd, vecSub, vecMulS, vecDivS, vecDotP, vecCrossP
    from vecMath import unitVec, vecMag, vecMagSq
    from vecMath import matDotV, xformMatRotZ
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
    
USE_LINEARIZED = False  # Use linearized or non-linearized eqom
USE_FEEDBACK   = False  # Compute accelerations using feedback forces & torques

T_STEP =  0.001  # Simulation and integration time step size (sec)
T_STOP =  0.080  # Simulation and integration stop time (sec)

PRINT_DATA = True   # Controls printing of collected data
PRINT_FBCK = True   # Controls printing of joint feedback
PRINT_EVAL = True   # Controls printing of dynamics evaluations
PLOT_DATA  = False  # Controls plotting of collected data
SAVE_ANIM  = False  # Controls saving animation images

# Drawing constants

WINDOW_RESOLUTION = (640, 480)

DRAW_SCALE = WINDOW_RESOLUTION[0] / 5
"""Factor to multiply physical coordinates by to obtain screen size in pixels"""

DRAW_OFFSET = (WINDOW_RESOLUTION[0] / 2, 50)
"""Screen coordinates (in pixels) that map to the physical origin (0, 0, 0)"""

BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR       = ( 10,  10,  10)
LINE_WIDTH       = 2  # width of the line (in pixels) representing a joint
LINE_COLOR       = ( 50,  50,  50)  # for drawing joint orientations from ODE
BODY_COLOR       = (  0,  50, 200)  # for drawing body positions from ODE
ROD_WIDTH        = 1  # width of the line (in pixels) representing a rod
ROD_COLOR        = (255,   0, 255)  # for drawing pendulum rod orientations from RK4  
MASS_COLOR       = (200,   0,  0)  # for drawing pendulum mass positions from RK4

# Double Pendulum System - characterization and initial conditions

"""
Pictogram of double pendulum system.


              origin (anchor for joint 1)
                V
    ------------o------------> +x
                |\
                | \
                |  \<- 1st massless rod
                |   \
                |    \
                |     \
                |      \
                |=(+T1)>\ T1 is joint 1 angle used in RK4 EQOM
                |        \
                V         O <- 1st mass (body 1 and anchor for joint 2)
               -y        /|.
                        / | .
                       /  -  .
                      /<(-P)==. P is joint 2 angle used in RK4 EQOM
                     /    -
          2nd rod ->/     |
                   /      |
                  /<(-T2)=|
                 /
                O <- 2nd mass (body 2)
                
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

ORIGIN  = (0.0, 0.0, 0.0)    # Cartesian coordinates of world space origin
Z_AXIS  = (0.0, 0.0, 1.0)    # direction vector of +rotation about Z axis
GRAVITY = (0.0, -9.81, 0.0)  # gravitational acceleration vector (m/sec/sec)

# Physical specifications

ROD_LENGTH  = 1.0            # in meters (m)
BODY_MASS   = 1.0            # in kilograms (kg)
BODY_RADIUS = 0.15           # in meters (m)
BODY_ROTDST = ROD_LENGTH
BODY_IZZc   = (2.0/5.0)*BODY_MASS*(BODY_RADIUS*BODY_RADIUS)
BODY_IZZj   = BODY_IZZc + BODY_MASS*(BODY_ROTDST*BODY_ROTDST)

MIN_PE = 3*BODY_MASS*GRAVITY[1]*ROD_LENGTH  # min system potential energy (J)

# Initial conditions

Theta0_1 = 30.0*RPD  # initial absolute angle (T1) of 1st rod attached to origin
Theta0_2 = 45.0*RPD  # initial absolute angle (T2) of 2nd rod attached to 1st rod

Theta0 = Theta0_1           # initial angle (T) of 1st rod attached to origin
Phi0   = Theta0_2 - Theta0  # initial angle (P) of 2nd rod attached to 1st rod
Tdot0  =  0.0*RPD           # initial dTheta/dt
Pdot0  =  0.0*RPD           # initial dPhi/dt

# Physical configuration

def rodAngleToXYZ(ang):
    """
    Calculates xyz coordinates of the rod end point given rotation ang in 
    radians measured clockwise positive from -y axis (refer to pictogram
    of double pendulum system presented above).
    """
    xyz = (ROD_LENGTH*sin(ang), -ROD_LENGTH*cos(ang), 0.0)
    return xyz

SPHERE1_POSITION = vecAdd(ORIGIN,rodAngleToXYZ(Theta0))
SPHERE1_ROTATION = xformMatRotZ(Theta0_1)
SPHERE1_MASS     = BODY_MASS
SPHERE1_RADIUS   = BODY_RADIUS
SPHERE1_COLOR    = BODY_COLOR

SPHERE2_POSITION = vecAdd(SPHERE1_POSITION,rodAngleToXYZ(Theta0 + Phi0))
SPHERE2_ROTATION = xformMatRotZ(Theta0_2)
SPHERE2_MASS     = BODY_MASS
SPHERE2_RADIUS   = BODY_RADIUS
SPHERE2_COLOR    = BODY_COLOR

JOINT1_ANCHOR = ORIGIN
JOINT1_AXIS   = Z_AXIS
JOINT1_COLOR  = LINE_COLOR
JOINT1_WIDTH  = LINE_WIDTH

JOINT2_ANCHOR = SPHERE1_POSITION
JOINT2_AXIS   = Z_AXIS
JOINT2_COLOR  = LINE_COLOR
JOINT2_WIDTH  = LINE_WIDTH

#============================================================================
# System of ordinary differential equations (ode) characterizing the motion
# of a double pendulum as described in problem 6-4 on page 276 and derived
# from the linearized equations given in part (b) for the solution to 6-4
# on page 505 of reference [1]. Specifically, the following two equations
#
#   m*l*l*(5*Tddot + 2*Pddot) + m*g*l*(3*T + P) = 0               (1)
#   m*l*l*(2*Tddot + Pddot) + m*g*l*(T + P)     = 0               (2)
#
# where
#
#   m     = mass of pendulum bodies (particles)
#   l     = length of pendulum massless rods
#   g     = gravitational acceleration
#   T     = theta
#   P     = phi
#   Tddot = d(dT/dt)/dt
#   Pddot = d(dP/dt)/dt
#
# can be reduced to the following two equations by dividing the left hand
# sides by the term "m*l*l".
#
#   5*Tddot + 2Pddot + (g/l)*(3*T + P) = 0                        (1a)
#   2*Tddot + Pddot + (g/l)*(T + P)    = 0                        (2a)
#
# These two equations can be solved for Tddot and Pddot yielding the
# following expressions.
#
#   Tddot = (g/l)*(P - T)   ==>  d(dT/dt)/dt = (g/l)*(phi - theta)
#   Pddot = (g/l)*(T -3*P)  ==>  d(dP/dt)/dt = (g/l)*(theta - 3*phi)
#
# The non-linear differential equations of motion given in part (a) for the
# solution to 6-4 on page 505 of reference [1] are rearranged in the form
# Y' = Ainv*G(y,y') as presented in the associated tutorial2.html file.
# Specifically, the matrices of the discrete state representation are as
# follows:
#
#        | d(dT/dt)/dt |
#   Y' = |             |
#        | d(dP/dt)/dt |
#
#          | -1           cos(P) + 1      |
#   Ainv = |                              | * (1/(cos(P)^2 - 2))
#          | cos(P) + 1   -(2*cos(P) + 3) |
#
#               | (g/l)*(sin(T+P) + 2*sin(T)) - sin(P)*(dP/dt)*(dP/dt + 2*(dT/dt)) |
#   G(y,y') = - |                                                                  |
#               | (g/l)*sin(T+P) + sin(P)*(dT/dt)^2                                |
#
#             | T |           | dT/dt |
#   where y = |   | and  y' = |       |
#             | P |           | dP/dt |
#
# References:
#
#   [1] Greenwood, Donald T., "Principles of Dynamics." Prentice-Hall,
#       Inc.: Englewood Cliffs, N.J., 1965.

nSvar = 5
S     = np.zeros(nSvar)
dS    = np.zeros(nSvar)
g     = vecMag(GRAVITY)
gdivl = g/ROD_LENGTH

S[0] = 0.0     # t0
S[1] = Theta0  # T
S[2] = Tdot0   # dT/dt
S[3] = Phi0    # P
S[4] = Pdot0   # dP/dt

def dotS(n,S):
    """
    State derivatives function.
    """
    global USE_LINEARIZED
    global gdivl
    
    dS    = np.zeros(n)
    dS[0] = 1.0                      # dt/dt
    
    if USE_LINEARIZED:
        # Refer to comment block above.
        dS[1] = S[2]                 # dT/dt
        dS[2] = gdivl*(S[3]-S[1])    # d(dT/dt)/dt
        dS[3] = S[4]                 # dP/dt
        dS[4] = gdivl*(S[1]-3*S[3])  # d(dP/dt)/dt
    else:
        # Refer to comment block above.
        #... compute Ainv
        cosP      = cos(S[3])
        Ainv      = np.zeros((2,2),dtype=np.float)
        Ainv[0,0] = -1.0
        Ainv[0,1] = cosP + 1.0
        Ainv[1,0] = Ainv[0,1]
        Ainv[1,1] = -(2*cosP + 3.0)
        Ainv      = Ainv/(cosP**2 - 2.0)
        #... compute G
        sinT   = sin(S[1])
        sinP   = sin(S[3])
        sinTpP = sin(S[1]+S[3])
        G      = np.zeros((2,1),dtype=np.float)
        G[0,0] = gdivl*(sinTpP + 2*sinT) - sinP*S[4]*(S[4] + 2*S[2])
        G[1,0] = gdivl*sinTpP + sinP*S[2]**2
        G      = -G
        #...  compute Y' = Ainv*G
        AinvG = Ainv.dot(G)
        #... load state vector values
        dS[1] = S[2]
        dS[2] = AinvG[0,0]
        dS[3] = S[4]
        dS[4] = AinvG[1,0]
        
    return dS

# Import procedures to compute accelerations from ODE joint and body states
# using equations of motion for compound double pendulum system dynamics.
    
from tutorial2eval import compute_wdot_vdot, calcFwdDynamicsODE

#============================================================================

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

from tutorial2util import printFeedback, bodyRotMatrix

#----------------------------------------------------------------------------
    
# Initialize pygame.
pygame.init()

# Open a display.
screen = pygame.display.set_mode(WINDOW_RESOLUTION)

# Create an ODE world object.
world = ode.World()
world.setGravity(GRAVITY)
world.setERP(1.0)
world.setCFM(0.0)

# Create fixed anchor body for double pendulum system.

body0 = ode.Body(world)
mass0 = ode.Mass()
mass0.setSphereTotal(BODY_MASS, BODY_RADIUS)
body0.setMass(mass0)
body0.setPosition(ORIGIN)
body0.setQuaternion((cos(0.0),0,0,sin(0.0)))

# Create two bodies for double pendulum system.

body1 = ode.Body(world)
mass1 = ode.Mass()
mass1.setSphereTotal(SPHERE1_MASS, SPHERE1_RADIUS)
body1.setMass(mass1)
body1.setPosition(SPHERE1_POSITION)
body1.setQuaternion((cos(Theta0_1/2),0,0,sin(Theta0_1/2)))

body2 = ode.Body(world)
mass2 = ode.Mass()
mass2.setSphereTotal(SPHERE2_MASS, SPHERE2_RADIUS)
body2.setMass(mass2)
body2.setPosition(SPHERE2_POSITION)
body2.setQuaternion((cos(Theta0_2/2),0,0,sin(Theta0_2/2)))

# Connect body0 with the static environment.
j0 = ode.FixedJoint(world)
j0.attach(body0, ode.environment)
j0.setFeedback(True)

# Connect body1 with body0.
j1 = ode.HingeJoint(world)
j1.attach(body1, body0)
j1.setAnchor(JOINT1_ANCHOR)
j1.setAxis(JOINT1_AXIS)
j1.setFeedback(True)

# Connect body2 with body1.
j2 = ode.HingeJoint(world)
j2.attach(body2, body1)
j2.setAnchor(JOINT2_ANCHOR)
j2.setAxis(JOINT2_AXIS)
j2.setFeedback(True)

# Define pygame circle radius for drawing each body sphere.
sph1_rad = int(DRAW_SCALE * SPHERE1_RADIUS)
sph2_rad = int(DRAW_SCALE * SPHERE2_RADIUS)

# Create background for text and clearing screen.
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(BACKGROUND_COLOR)

# Write title.
if pygame.font:
    font = pygame.font.Font(None, 36)
    if USE_LINEARIZED:
        text = font.render("PyODE Tutorial 2 - Dble Pendulum (linearized)", 1, TEXT_COLOR)
    else:
        text = font.render("PyODE Tutorial 2 - Dble Pendulum (non-linear)", 1, TEXT_COLOR)
    textpos = text.get_rect(centerx=background.get_width()/2)
    background.blit(text, textpos)
    font = pygame.font.Font(None, 24)
    
# Clear screen.
screen.blit(background, (0, 0))
pygame.display.flip()

# Simulation loop...

FPS    = 25.0
F_TIME = 1/FPS
N_STEP = int(floor((F_TIME + 0.5*T_STEP)/T_STEP))
N_TIME = N_STEP*T_STEP

if __name__ == "__main__":
    
    # Instantiate clock to regulate display updates. 
    
    clk = pygame.time.Clock()
    
    # Create simulation data collection arrays for plotting.
    
    nSamples = int(ceil(T_STOP/N_TIME)) + 1
    
    if PLOT_DATA:
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
        TOTErk4 = np.zeros(nSamples)  # total system enerrgy from RK4
        WDOTode = np.zeros((nSamples,2))  # Angular accelerations from ODE
        WDOTrk4 = np.zeros((nSamples,2))  # Angular accelerations from RK4
    
    # Take a very small step in ODE to set rates and feedback data.
    
    world.step(T_STEP/1000.0)
    
    # Instantiate a temporary Runge-Kutta 4th order ode solver, 
    # initialize, take same very small step to match ODE and
    # save system state from step, then delete the RK4 solver 
    # object.
    
    rk4temp = RK4_Solver(T_STEP/1000.0,nSvar)
    
    rk4temp.init(S)
    
    S = rk4temp.step(S,dotS)
    
    del rk4temp
        
    # Instantiate Runge-Kutta 4th order ode solver, initialize
    # using current state with state time set to zero.
    
    rk4 = RK4_Solver(T_STEP,nSvar)
    
    S[0] = 0.0
    
    rk4.init(S)

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
            text      = font.render("Simulation Time = %8.3f (sec)" % t, 1, TEXT_COLOR)
            textpos   = text.get_rect(centerx=background.get_width()/2)
            textpos.y = background.get_height() - int(ceil(1.5*textpos.height))
            screen.blit(text, textpos)
        
        # Get current ODE joint and body world space positions.
        j1Pos = j1.getAnchor()
        j2Pos = j2.getAnchor()
        b0    = j0.getBody(0)
        b1    = j1.getBody(0)
        b2    = j2.getBody(0)
        b0Pos = b0.getPosition()
        b1Pos = b1.getPosition()
        b2Pos = b2.getPosition()
        
        # Get current body to world rotation matrices.
        b1Rot = bodyRotMatrix(b1)
        b2Rot = bodyRotMatrix(b2)
        
        # Get current double pendulum mass world space positions.
        m1Pos = vecAdd(rodAngleToXYZ(S[1]),ORIGIN)
        m2Pos = vecAdd(rodAngleToXYZ(S[1]+S[3]),m1Pos)
        
        # Draw the two bodies and the lines representing the joints.
        pygame.draw.line(screen, JOINT1_COLOR, coord(j1Pos), coord(b1Pos), JOINT1_WIDTH)
        pygame.draw.line(screen, JOINT2_COLOR, coord(j2Pos), coord(b2Pos), JOINT2_WIDTH)
        pygame.draw.circle(screen, SPHERE1_COLOR, coord(b1Pos, integer=True), sph1_rad, 0)
        pygame.draw.circle(screen, SPHERE2_COLOR, coord(b2Pos, integer=True), sph2_rad, 0)
        
        # Draw the double pendulum composed of two masses and the connecting rods.
        pygame.draw.line(screen, ROD_COLOR, coord(ORIGIN), coord(m1Pos), ROD_WIDTH)
        pygame.draw.line(screen, ROD_COLOR, coord(m1Pos), coord(m2Pos), ROD_WIDTH)
        pygame.draw.circle(screen, MASS_COLOR, coord(m1Pos, integer=True), sph1_rad, 1)
        pygame.draw.circle(screen, MASS_COLOR, coord(m2Pos, integer=True), sph2_rad, 1)
        
        # Display updated screen.
        pygame.display.flip()
        if SAVE_ANIM:
            istr = format_string("%04d", i)
            fpth = "./anim/tutorial2x_" + istr + ".png"
            pygame.image.save(screen, fpth)
            
        # Get body linear and angular velocities.
        b0LVel = b0.getLinearVel()
        b0AVel = b0.getAngularVel()
        b1LVel = b1.getLinearVel()
        b1AVel = b1.getAngularVel()
        b2LVel = b2.getLinearVel()
        b2AVel = b2.getAngularVel()
        
        # Get joint rotation angle and angle rates.
        j1angle = j1.getAngle()
        j1omega = j1.getAngleRate()
        j2angle = j2.getAngle()
        j2omega = j2.getAngleRate()
        j0AVel  = b0AVel
        j1AVel  = vecSub(b1AVel,b0AVel)  # same as j1omega*j1.getAxis()
        j2AVel  = vecSub(b2AVel,b1AVel)  # same as j2omega*j2.getAxis()
        
        # Convert joint angles and rates to absolute (refer to pictogram notes).
        j1theta = Theta0_1 + j1angle
        j2theta = Theta0_2 + j1angle + j2angle
        j1thdot = j1omega
        j2thdot = j2omega + j1omega
    
        # Get joint feedback forces and torques.
        fb1 = j1.getFeedback()
        fb2 = j2.getFeedback()
        if PRINT_FBCK:
            print("t : %7.3f" % t)
            printFeedback('j1',t,fb1)
            printFeedback('j2',t,fb2)
        
        # Assume total forces and torques on body are accounted for in feedback.
        b1Ftot = vecAdd(fb1[0],fb2[2])
        b1Ttot = vecAdd(fb1[1],fb2[3])
        b2Ftot = fb2[0]
        b2Ttot = fb2[1]

        # Calculate accelerations of ODE joints and bodies.
        if USE_FEEDBACK:
            # use joint feedback torques and forces on body COM
            alpha1 = vecDivS(b1Ttot,BODY_IZZc)[2]
            alpha2 = vecDivS(b2Ttot,BODY_IZZc)[2]
            wdot = np.zeros((3,1))  # rotational acceleration of joints
            wdot[1,0] = alpha1
            wdot[2,0] = alpha2
            accel1 = vecDivS(vecAdd(b1Ftot,GRAVITY),b1.getMass().mass)
            accel2 = vecDivS(vecAdd(b2Ftot,GRAVITY),b2.getMass().mass)
            vdot = np.zeros((3,3))  # linear acceleration of bodies
            vdot[1,0:3] = accel1[0:3]
            vdot[2,0:3] = accel2[0:3]
        else:
            # use compound double pendulum dynamics equations
            (wdot,vdot) = compute_wdot_vdot(j1, Theta0_1, j2, Theta0_2, g)
        
        # Calculate angular accelerations of RK4 joints.
        dS     = dotS(nSvar,S)
        t1dot  = dS[1]
        t1ddot = dS[2]
        t2dot  = t1dot + dS[3]
        t2ddot = t1ddot + dS[4]  # Note: Vectorially t1dot x dS[3] is zero since
                                 #       both are collinear with Z world axis
                                 #       (see eq 6.31 on pg 173 of ref [6]).
        if PLOT_DATA: 
            WDOTode[i,0] = wdot[1]
            WDOTode[i,1] = wdot[2]
            WDOTrk4[i,0] = t1ddot
            WDOTrk4[i,1] = t2ddot
            
        # Calculate linear accelerations of ODE bodies using Articulated
        # -Body forward dynamics algorithm.
        results = calcFwdDynamicsODE([j0,j1,j2],Theta0_1,Theta0_2,g,PRINT_EVAL)
        (Pj,rth,ath,w,a,Pb,Vb,Wb,Ab,jtau) = results
        
        p1ddot = Ab[1]
        p2ddot = Ab[2]
        
        # Calculate forces and moments on ODE bodies using calculated
        # body linear accelerations and joint angular accelerations with
        # ODE body angular velocities and compare results with forces and
        # torques from ODE joint feedback.
        # F = m*vdot - m*G
        frc1 = vecSub(vecMulS(p1ddot,b1.getMass().mass), \
                      vecMulS(GRAVITY,b1.getMass().mass))
        frc2 = vecSub(vecMulS(p2ddot,b2.getMass().mass), \
                      vecMulS(GRAVITY,b2.getMass().mass))
        # M = Ic*wdot + w x (Ic*w)
        mom1 = vecAdd(matDotV(b1.getMass().I,vecMulS(j1.getAxis(),wdot[1])), \
                      vecCrossP(j1AVel,matDotV(b1.getMass().I,j1AVel)))
        mom2 = vecAdd(matDotV(b2.getMass().I,vecMulS(j2.getAxis(),wdot[2])), \
                      vecCrossP(j2AVel,matDotV(b2.getMass().I,j2AVel)))
        if PRINT_FBCK:
            print("j1:  w1dot = %8.4f  t1ddot = %8.4f" % \
                  (wdot[1],t1ddot) )
            print("j2:  w2dot = %8.4f  t2ddot = %8.4f" % \
                  (wdot[2],t2ddot) )
            print("b1:  vdot = %8.4f  p1ddot = %8.4f  frc1 = %8.4f  mom1 = %8.4f  b1Ftot = %8.4f  b1Ttot = %8.4f" % \
                  (vecMag(vdot[1]),vecMag(p1ddot),vecMag(frc1),vecMag(mom1),vecMag(b1Ftot),vecMag(b1Ttot)) )
            print("b2:  vdot = %8.4f  p2ddot = %8.4f  frc2 = %8.4f  mom2 = %8.4f  b2Ftot = %8.4f  b2Ttot = %8.4f" % \
                  (vecMag(vdot[2]),vecMag(p2ddot),vecMag(frc2),vecMag(mom2),vecMag(b2Ftot),vecMag(b2Ttot)) )
        
        # Calculate joint angular accelerations using calculated moments and 
        # body linear accelerations.
        alpha1 = vecAdd(vecDivS(mom1,BODY_IZZj), \
                        vecDivS(vecCrossP(vecSub(b1Pos,j1Pos), \
                                          vecMulS(p1ddot,b1.getMass().mass)), \
                                BODY_IZZj))[2]
        alpha2 = vecDivS(mom2,BODY_IZZc)[2]
                
        # Calculate tension force along rod from body to joint.
        r1Uvec = unitVec(vecSub(j1Pos,b1Pos))
        b1Frod = vecDotP(vecMulS(p1ddot,b1.getMass().mass),r1Uvec) \
               + BODY_MASS*vecDotP(vecCrossP(vecMulS(j1.getAxis(),wdot[1]),vecSub(b1Pos,j1Pos)),r1Uvec) \
               + BODY_MASS*vecDotP(vecCrossP(b1AVel,b1LVel),r1Uvec) \
               - BODY_MASS*vecDotP(GRAVITY,r1Uvec)
        b1Frod = vecDotP(vecAdd(vecMulS(vecSub(p1ddot,GRAVITY),b1.getMass().mass), \
                                vecMulS(vecSub(p2ddot,GRAVITY),b2.getMass().mass)),r1Uvec)
        r2Uvec = unitVec(vecSub(j2Pos,b2Pos))
        b2Frod = vecDotP(vecMulS(p2ddot,b2.getMass().mass),r2Uvec) \
               + BODY_MASS*vecDotP(vecCrossP(vecMulS(j2.getAxis(),wdot[2]),vecSub(b2Pos,j2Pos)),r1Uvec) \
               + BODY_MASS*vecDotP(vecCrossP(b2AVel,b2LVel),r2Uvec) \
               - BODY_MASS*vecDotP(GRAVITY,r2Uvec)
        b2Frod = vecDotP(vecMulS(vecSub(p2ddot,GRAVITY),b2.getMass().mass),r2Uvec)       
        if PRINT_FBCK:
            print("j1:  Tdot = %8.4f  Tddot = %8.4f  alpha1 = %8.4f  Frod = %8.3f (%8.3f)" % \
                  (t1dot,t1ddot,alpha1,b1Frod,vecDotP(fb1[0],r1Uvec)) )
            print("j2:  Tdot = %8.4f  Tddot = %8.4f  alpha2 = %8.4f  Frod = %8.3f (%8.3f)" % \
                  (t2dot,t2ddot,alpha2,b2Frod,vecDotP(fb2[0],r2Uvec)) )
            
        # Collect data for printing/plotting.
        # ... time steps
        if PRINT_DATA and not PRINT_FBCK: 
            print("t : %7.3f" % t)
        if PLOT_DATA:
            Time[i] = t
        # ... linear and angular velocity for body 1 from ODE and RK4
        if PRINT_DATA:
            print("j1:  angle (deg) = %8.3f  omega (rad/sec) = %8.3f" % \
                  (j1angle*DPR, j1omega) )
            print("b1:  LVel = %8.3f %8.3f %8.3f  AVel = %8.3f %8.3f %8.3f" % \
                (b1LVel[0],b1LVel[1],b1LVel[2],b1AVel[0],b1AVel[1],b1AVel[2]) )
        Velode = vecCrossP(b1AVel,vecSub(b1Pos,j1Pos)) 
        Velrk4 = vecCrossP(vecMulS(JOINT1_AXIS,S[2]),m1Pos)
        m1LVel = Velrk4
        if PRINT_DATA:
            print("b1:  Velode = %8.3f %8.3f %8.3f" % \
                  (Velode[0],Velode[1],Velode[2]) )
            print("m1:  Velrk4 = %8.3f %8.3f %8.3f" % \
                  (Velrk4[0],Velrk4[1],Velrk4[2]) )
        if PLOT_DATA:
            B1LVode[i] = vecMag(Velode)  # Velode should be same as b1LVel
            B1AVode[i] = j1omega*DPR     # j1omega should be same as b1AVel[2] 
            B1LVrk4[i] = vecMag(Velrk4)
            B1AVrk4[i] = S[2]*DPR
        # ... linear and angular velocity for body 2 from ODE and RK4
        if PRINT_DATA:
            print("j2:  angle (deg) = %8.3f  omega (rad/sec) = %8.3f" % \
                  (j2angle*DPR, j2omega) )
            print("b2:  LVel = %8.3f %8.3f %8.3f  AVel = %8.3f %8.3f %8.3f" % \
                (b2LVel[0],b2LVel[1],b2LVel[2],b2AVel[0],b2AVel[1],b2AVel[2]) )
        Velode = vecAdd(vecCrossP(b2AVel,vecSub(b2Pos,j2Pos)),Velode)
        angVel = vecMulS(JOINT2_AXIS,(S[2]+S[4]))
        Velrk4 = vecAdd(vecCrossP(angVel,vecSub(m2Pos,m1Pos)),Velrk4)
        m2LVel = Velrk4
        if PRINT_DATA:
            print("b2:  Velode = %8.3f %8.3f %8.3f" % \
                  (Velode[0],Velode[1],Velode[2]) )
            print("m2:  Velrk4 = %8.3f %8.3f %8.3f" % \
                  (Velrk4[0],Velrk4[1],Velrk4[2]) )
        if PLOT_DATA:
            B2LVode[i] = vecMag(Velode)         # Velode should be same as b2LVel
            B2AVode[i] = (j1omega+j2omega)*DPR  # j1omega+j2omega should be same as b2AVel[2]
            B2LVrk4[i] = vecMag(Velrk4)
            B2AVrk4[i] = (S[2]+S[4])*DPR
        #
        # Note: The following calculations of total energy for the RK4 integrated
        #       differential equations of motion will only be an approximation of
        #       the double pendulum system modeled as point masses.
        #
        # ... translational kinetic energy
        #     refer to eqs. (2.5) and (2.6) on pg. 4 of reference [2]
        b1KEtrn = 0.0  # CoM of body 1 only rotates about joint 1
        b2KEtrn = 0.5*BODY_MASS*vecMagSq(b2LVel)
        m1KEtrn = 0.0  # CoM of mass 1 only rotates about joint 1
        m2KEtrn = 0.5*BODY_MASS*vecMagSq(m2LVel)
        # ... rotational kinetic energy (not directly applicable to point masses)
        #     refer to eqs. (2.5) and (2.6) on pg. 4 of reference [2]
        b1KErot = 0.5*BODY_IZZj*vecMagSq(b1AVel)
        b2KErot = 0.5*BODY_IZZc*vecMagSq(b2AVel)
        m1KErot = 0.5*BODY_IZZj*S[2]**2
        m2KErot = 0.5*BODY_IZZc*(S[2]+S[4])**2
        # ... potential energy (relative to origin at joint 1 anchor)
        #     refer to eq. (2.13) on pg. 5 of reference [2]
        b1PE = BODY_MASS*g*b1Pos[1]
        b2PE = BODY_MASS*g*b2Pos[1]
        m1PE = BODY_MASS*g*m1Pos[1]
        m2PE = BODY_MASS*g*m2Pos[1]
        # ... total system energy (relative to minimum potential energy)
        TEode = b1KEtrn + b2KEtrn + b1KErot + b2KErot + b1PE + b2PE - MIN_PE
        TErk4 = m1KEtrn + m2KEtrn + m1KErot + m2KErot + m1PE + m2PE - MIN_PE
        if PRINT_DATA:
            print("totEode = %10.4f" % TEode)
            print("totErk4 = %10.4f" % TErk4)
        if PLOT_DATA:    
            TOTEode[i] = TEode
            TOTErk4[i] = TErk4
        
        # Next simulation steps.
        for n in range(N_STEP):
            world.step(T_STEP)
            S = rk4.step(S,dotS)
            
        # Increment simulation time and data sample index.
        t = t + N_TIME
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
            t = t + T_STEP
            i = i + 1
    
        # Create and show the plots.
        
        if USE_LINEARIZED:
            figdir = "./imgs/1a/" 
        else:
            figdir = "./imgs/1b/"

        figures = []
            
        figures.append(plt.figure(1, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 1 ||Lin Vel|| for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Absolute Linear Velocity (m/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B1LVode,'b-',Time,B1LVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='upper left')
        plt.savefig(figdir+"Figure_1.png", format='png')
    
        figures.append(plt.figure(2, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 1 Ang Vel for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Velocity (deg/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B1AVode,'b-',Time,B1AVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='upper left')
        plt.savefig(figdir+"Figure_2.png", format='png')

        figures.append(plt.figure(3, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 2 ||Lin Vel|| for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Absolute Linear Velocity (m/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B2LVode,'b-',Time,B2LVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='upper left')
        plt.savefig(figdir+"Figure_3.png", format='png')

        figures.append(plt.figure(4, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum Body 2 Ang Vel for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Velocity (deg/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,B2AVode,'b-',Time,B2AVrk4,'r:2')
        plt.legend(('ODE','RK4'),loc='upper left')
        plt.savefig(figdir+"Figure_4.png", format='png')

        figures.append(plt.figure(5, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum System Total Energy for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Energy wrt Minimum Potential (Joules)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,TOTEode,'b-',Time,TOTErk4,'r:2')
        plt.legend(('ODE','RK4'),loc='lower left')
        plt.savefig(figdir+"Figure_5.png", format='png')

        figures.append(plt.figure(6, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum System Joint 1 Wdot for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Acceleration (rad/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,WDOTode[0:,0],'b-',Time,WDOTrk4[0:,0],'r:2')
        plt.legend(('ODE','RK4'),loc='upper left')
        plt.savefig(figdir+"Figure_6.png", format='png')

        figures.append(plt.figure(7, figsize=(8,6), dpi=80))
        plt.title("Double Pendulum System Joint 2 Wdot for ODE and RK4")
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Acceleration (rad/sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,WDOTode[0:,1],'b-',Time,WDOTrk4[0:,1],'r:2')
        plt.legend(('ODE','RK4'),loc='upper left')
        plt.savefig(figdir+"Figure_7.png", format='png')

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
    
