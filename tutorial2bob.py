#!/usr/bin/env ipython --matplotlib=qt

# pylint: disable=trailing-whitespace,bad-whitespace,invalid-name,anomalous-backslash-in-string

# File: tutorial2bob.py
# Auth: Gary E. Deschaines
# Date: 19 May 2015
# Prog: Single pendulum system modeled with PyODE, animated with Pygame
# Desc: Models numerical solution for single bob pendulum system dynamics 
#       originally presented as example 2 in PyODE tutorials.
#
#         http://pyode.sourceforge.net/tutorials/tutorial2.html
#
# pyODE example 2: Connecting bodies with joints
#
# modified by Gideon Klompje (removed literals and using
# 'ode.Mass.setSphereTotal' instead of 'ode.Mass.setSphere')
#
# modified by Gary E. Deschaines (changed double planar
# pendulum to single mass, ball joint to hinge, and uses 
# matplotlib to create plots of data collected from the 
# ODE simulation and results calculated from body dynamics 
# evaluation)
#
# References:
#
#   [1] Greenwood, Donald T., "Principles of Dynamics." Prentice-Hall,
#       Inc.: Englewood Cliffs, N.J., 1965.
#
#   [2] Awrejcewicz, J., "Classical Mechanics: Dynamics." Springer-
#       Verlag: New York, N.Y., 2012.
#
# Disclaimer:
#
# See DISCLAIMER

import sys

from math import ceil, floor, atan2, cos, sin, sqrt
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
    from vecMath import vecAdd, vecDotP, vecCrossP, vecDivS
    from vecMath import vecMag, vecMagSq, vecMulS, vecSub
    from vecMath import unitVec, acosUVecDotP
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
    
# Pendulum system parameters and output control flags
  
THETA_0  = 90.0*RPD  # initial pendulum angle (rad)
MASS     = 1.0       # of plumb bob (kg)
LENGTH   = 1.0       # of rod from pivot joint to plumb bob CoM (m)

T_STEP   = 0.001     # Simulation time step size (sec)
T_STOP   = 2.368     # Simulation stop time (sec) 
                     # Note: from page 118 of ref [1], for Theta0 of 90 
                     # degrees the period of oscillation is given by
                     # 7.4164*sqrt(LENGTH/G).

PRINT_DATA = True    # Controls printing of calculation data and results
PLOT_DATA  = True    # Controls plotting of calculation data and results
SAVE_ANIM  = False   # Controls saving animation images

# Drawing constants

WINDOW_RESOLUTION = (640, 480)

DRAW_SCALE = WINDOW_RESOLUTION[0] / max(2*LENGTH,1)
"""Factor to multiply physical coordinates by to obtain screen size in pixels"""

DRAW_OFFSET = (WINDOW_RESOLUTION[0] / 2, 50)
"""Screen coordinates (in pixels) that map to the physical origin (0, 0, 0)"""

FILL_COLOR = (255, 255, 255)  # background fill color
TEXT_COLOR = ( 10,  10,  10)  # text drawing color
J_COLOR    = (100, 100, 100)  # for drawing joint orientation
J_LWID     = 2   # width of drawn line (in pixels) representing the joint
B_COLOR    = (50, 0, 200)  # for drawing body position
B_CRAD     = 15  # radius of drawn circle (in pixels) representing the body
X_COLOR    = (200, 0, 50)  # drawn for extrapolated body position

# Utility functions

def coord(Wxyz, integer=False):
    """
    Convert world coordinates to pixel coordinates.  Setting 'integer' to
    True will return integer coordinates.
    """
    xs = (DRAW_OFFSET[0] + DRAW_SCALE*Wxyz[0])
    ys = (DRAW_OFFSET[1] - DRAW_SCALE*Wxyz[1])

    if integer:
        return int(round(xs)), int(round(ys))
    
    return xs, ys
      
def distAngleToXYZ(dist,ang):
    """ Converts given distance and angle (in radians) to xyz coordinate.
    """
    xyz = (dist*sin(ang), -dist*cos(ang), 0.0)
    return xyz
    
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

def rotToM(R):
    """
    Converts given ODE rotation 9-tuple matrix to row vector matrix
    form used by the vecMath matrix functions.
    """
    mat0 = ( R[0], R[1], R[2] )
    mat1 = ( R[3], R[4], R[5] )
    mat2 = ( R[6], R[7], R[8] )
    return (mat0, mat1, mat2)
    
def printVec(name,V):
    """
    Print name, components and magnitude of given vector.
    """
    fmt = "%6s = %8.4f %8.4f %8.4f  %9.4f"
    print(fmt % (name, V[0], V[1], V[2], vecMag(V)) )
    
# Single Compound Planar Pendulum System - characterization and 
#                                          initial conditions

"""
Pictogram of the single compound planar (1 DoF) pendulum system.


              origin (anchor for joint 1)
                |
                |  Inertial Reference Frame
                V
    ------------o------------> +X axis
                |\     "
                | \    "---{hinge joint angle
                |  \ (-a)
                |   \  "            Note: The +y axis of the body frame will
                |    \ V                  generally be pointed toward the
                |     \   +y              joint anchor, which corresponds to
                |      \   ^              a body rotation angle of (+T) and
                |       \  |              a hinge joint angle (-a) such that
                |==(+T)=>\ |              (T) is equal to (a) plus THETA_0,
                |   ^     \|              the rod angle when the hinge angle
                V   :      O----->+x      is at zero. 
            -Y axis :      ^   
                    :      :  Body Frame
                    :      :
         rod angle}-:      :-{CoM of pendulum bob mass
"""

# Physical constants
G       = 9.81            # gravitational acceleration (m/sec/sec)
GRAVITY = (0, -G, 0)      # gravitational acceleration vector
MIN_PE  = -MASS*G*LENGTH  # minimum system potential energy (J)

# Pendulum specifications
BALL   = 1
DISK   = 2
SHAPE  = BALL  # of plumb bob
RADIUS = 0.15  # of plumb bob

# Pivot joint
J_ANCHOR = (0, 0,  0)  # at origin of the inertial reference frame
J_AXIS   = (0, 0, -1)  # negative hinge angles are clockwise, as 
                       # shown in the pictogram above   

# Plumb bob mass characteristics
B_TANG = (1, 0, 0)  # tangential +x axis
B_NORM = (0, 1, 0)  # normal +y axis -- initially pointed toward joint
B_AXIS = (0, 0, 1)  # rotational +z axis
B_MASS = MASS
B_RAD  = RADIUS
B_WID  = 0.01
B_POS  = vecAdd(J_ANCHOR,distAngleToXYZ(LENGTH,THETA_0))
# moment of inertia about CoM
if SHAPE == BALL: B_Izzc = (2.0/5.0)*B_MASS*B_RAD**2
else:             B_Izzc = (1.0/2.0)*B_MASS*B_RAD**2
# moment of inertia about pivot joint (by parallel-axis theorem)
B_Izzj = B_Izzc + B_MASS*vecMagSq(vecSub(B_POS,J_ANCHOR))

B_Icom = ((B_Izzc,0,0), (0,B_Izzc,0), (0,0,B_Izzc))
B_Icor = ((B_Izzj,0,0), (0,B_Izzc,0), (0,0,B_Izzj))
B_Iinv = ((1/B_Izzj,0,0), (0,1/B_Izzc,0), (0,0,1/B_Izzj)) 

#-----------------------------------------------------------------------------
# State variables, derivatives arrays and derivatives function.

nSvar = 7                # number of state variables
S     = np.zeros(nSvar)  # state variables
dS    = np.zeros(nSvar)  # state derivatives
    
def dotS(n,S):
    """
    State derivatives function which incorporates the non-linear
    2nd order differential equations of motion for the modeled 
    physical planar pendulum system. The integration of this 
    function yields the position and rotation of the pendulum 
    body mass independent of the rigid-body dynamic solutions 
    generated by PyODE.
    """
    global G, GRAVITY, LENGTH, THETA_0
    global J_ANCHOR, B_AXIS, B_NORM, B_MASS, B_Izzc, B_Izzj
  
    dS = np.zeros(n)
    
    # Get current joint and body states.
    
    # joint anchor position (constant).
    Pj = J_ANCHOR              
    # body translation states.
    Pb = (S[1], S[2], 0.0)    # position (integrated from Vref)
    Vb = (S[3], S[4], 0.0)    # velocity (integrated from Aref)
    # joint/body rotation states.
    theta = S[5]              # angle (integrated from W)
    W     = (0.0, 0.0, S[6])  # rate (integrated from Wdot)
    
    #
    # Calculate the body's translational and rotational velocities
    # and accelerations from total forces and torques applied to 
    # the body.
    #
    # Notes:
    #
    #   {1} The pendulum rod angle theta is essentially the 
    #       one degree of freedom (1 DoF) which needs to be 
    #       determined by integrating the angular velocity 
    #       and acceleration of the body mass. However, the 
    #       derived X and Y components of the body's linear 
    #       inertial velocity are simultaneosly integrated 
    #       to yield the body's 2D position coordinates.
    #
    #   {2} Although not used to solve for w or theta, the 
    #       following relation exists for a physical pendulum 
    #       system with initial conditions theta(t) = theta0 
    #       and w(t) = 0 at time t = 0:  
    #
    #                2*m*g*d
    #         w^2 = --------- * [cos(theta)-cos(theta0)]
    #                  Io
    #
    #       where d is the distance from pivot point to body 
    #       CoM, and Io is the moment of inertia about the 
    #       pivot point. [1 pp 116-120,339-341], [2 pp 69-76,
    #       80-83]
    #
    #   {3} In this planar system the body's moment of inertia 
    #       with respect to the inertial reference frame is not 
    #       affected by rotations strictly about the body frame 
    #       z axis aligned with the reference frame Z axis.
    #
    # References:
    #   
    #   [1] Greenwood, Donald T., "Principles of Dynamics." Prentice-Hall, 
    #       Inc.: Englewood Cliffs, N.J., 1965.
    #
    #   [2] Awrejcewicz, J., "Classical Mechanics: Dynamics." Springer-
    #       Verlag: New York, N.Y., 2012.
    
    #... get body location and distance wrt to joint as
    #    measured in the inertial reference frame.
    Ploc = vecSub(Pb,Pj)
    dstP = vecMag(Ploc)
    
    #... get joint location and distance wrt to body as
    #    measured in the body frame.
    Jloc   = vecRotZ(vecSub(Pj,Pb),theta)
    dstJ   = vecMag(Jloc)
    dstJsq = vecMagSq(Jloc)
    
    #... get angle to direction of joint location wrt to
    #    the body frame +y axis, which should coincide 
    #    with the direction of the force applied to the
    #    joint by the rotating body.
    Jdir = unitVec(Jloc)
    psi  = acosUVecDotP(Jdir,B_NORM)
    jang = theta + psi
    
    #... body velocity in reference frame due to rotation
    #    rate about the joint axis, which is constrained
    #    to be the same as that due to the body's rotation
    #    about its rotation axis aligned with the joint axis.
    Vref = vecCrossP(W,Ploc)

    #... ensure velocity vector +x and +y components are 
    #    aligned with the tangential and normal force
    #    directions in the body frame (i.e., normal to
    #    and along vector from body's CoM to the joint). 
    Vbdy = vecRotZ(Vref,jang)
    
    #... get gravity components normal to and along vector
    #    from body's CoM to the joint.
    Agbj = vecRotZ(GRAVITY,jang)
    
    #... centripetal acceleration along vector from CoM 
    #    to joint due to angular velocity which acts 
    #    normal to the velocity vector resultant from
    #    rotation (note: W x Vbdy yields vector in the 
    #    direction of the body frame +y axis). This 
    #    acceleration contributes to the force at the
    #    joint which counters the tension/compression 
    #    in the rod exterted by the centripetal force 
    #    of the mass rotating about the joint. For the
    #    pendulum system, this acceleration plus the
    #    gravity component along the vector from CoM
    #    to joint must be propagated in order for the 
    #    joint anchor to remain stationary. Therefore
    #    the first force in the feedback data structure 
    #    for the joint (i.e., fb[0]) includes the sum
    #    of m*Awn and m*g*cos(theta).
    #      
    Awn = vecCrossP(W,Vbdy)
    
    #... angular acceleration associated with torque 
    #    at the joint caused by gravity when the body 
    #    CoM is offset from the -Y axis of the inertial
    #    reference frame (i.e., the force of gravity is 
    #    not aligned with the vector from the body CoM 
    #    to the joint anchor point). This torque is the
    #    rate of change in angular momentum d(Iw)/dt at
    #    the joint expressed as Izzj*Wdot and is equal
    #    to Ploc x Fg in the inertial reference frame,
    #    which is equivalent to Fg x Jloc in the body
    #    frame, or simply -m*g*sin(theta)*dstJ in this 
    #    planar system (also see Note {3} above).
    Izzj = B_Izzc + B_MASS*dstJsq      # parallel-axis theorem
    wdot = (B_MASS*Agbj[0]*dstJ)/Izzj  # Agbj[0] is -G*sin(theta)
    Wdot = vecMulS(B_AXIS, wdot) 
    
    #... acceleration normal to vector from CoM to joint
    #    due to angular acceleration which acts tangential 
    #    with the velocity vector resultant from the body's 
    #    rotation (note: Jloc x Wdot yields vector in the 
    #    direction of the body frame +x axis).
    Awt = vecCrossP(Jloc,Wdot)
    Awt = vecMulS(Awt,Izzj/(B_MASS*dstJsq))
    
    #... total acceleration in reference frame (inertial plus 
    #    applied gravity).
    Aref = vecRotZ(vecAdd(vecAdd(Awt,Awn),Agbj),-jang)
    
    # Update state derivatives array.
    
    dS[0] = 1.0      # dt/dt
    #... for body translational states (see Note {1} above).
    dS[1] = Vref[0]  # vx -+=> Pb(t+dt) = Pb(t) + Vref*dt with
    dS[2] = Vref[1]  # vy /     Vref = W x Pdst
    dS[3] = Aref[0]  # ax -+=> Vb(t+dt) = Vb(t) + Aref*dt with
    dS[4] = Aref[1]  # ay /     Aref = Wdot x Pdst + W x Vref + G
    #... for joint/body rotational states (see Note {2} above).
    dS[5] = W[2]     # w ====> theta(t+dt) = theta(t) + w*dt
    dS[6] = wdot     # wdot => w(t+dt) = w(t) + wdot*dt with
                     #           wdot = -m*g*sin(theta)*dstP/Izzj
    return dS
    
#-----------------------------------------------------------------------------

# Initialize pygame
pygame.init()

# Open a display
screen = pygame.display.set_mode(WINDOW_RESOLUTION)

# Create a world object
world = ode.World()
world.setGravity(GRAVITY)
world.setERP(1.0)
world.setCFM(0.0)

# Create body
body   = ode.Body(world)
mass_b = ode.Mass()
if SHAPE == BALL: mass_b.setSphereTotal(B_MASS, B_RAD)
else:             mass_b.setCylinderTotal(B_MASS, 3, B_RAD, B_WID) 
body.setMass(mass_b)
body.setPosition(B_POS)
body.setQuaternion((cos(THETA_0/2),0,0,sin(THETA_0/2)))

# Connect bob with the static environment using hinge joint
jh = ode.HingeJoint(world)
jh.attach(ode.environment,body)
jh.setAxis(J_AXIS)
jh.setAnchor(J_ANCHOR)
jh.setFeedback(True)
    
# Create background for text and clearing screen
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(FILL_COLOR)

# Write title
if pygame.font:
    title   = "PyODE Tutorial 2 - Physical Planar Pendulum Sim"
    font    = pygame.font.Font(None, 30)
    text    = font.render(title, 1, TEXT_COLOR)
    textpos = text.get_rect(centerx=background.get_width()/2)
    background.blit(text, textpos)
    font = pygame.font.Font(None, 24)
    
# Clear screen
screen.blit(background, (0, 0))

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
        Ft_Fbck = np.zeros(nSamples)  # Tangential force from feedback
        Fn_Fbck = np.zeros(nSamples)  # Normal force from feedback
        At_Fbck = np.zeros(nSamples)  # Tangential accel from feedback
        An_Fbck = np.zeros(nSamples)  # Normal accel from feedback
        Ft_Calc = np.zeros(nSamples)  # Tangential force from calculations
        Fn_Calc = np.zeros(nSamples)  # Normal force from calculations
        At_Calc = np.zeros(nSamples)  # Tangential accel from calculations
        An_Calc = np.zeros(nSamples)  # Normal accel from calculations
        Ft_Solv = np.zeros(nSamples)  # Tangential force from RK4 solver
        Fn_Solv = np.zeros(nSamples)  # Normal force from RK4 solver
        At_Solv = np.zeros(nSamples)  # Tangential accel from RK4 solver
        An_Solv = np.zeros(nSamples)  # Normal accel from RK4 solver
        Py_Fbck = np.zeros(nSamples)  # Y position from feedback
        Py_Calc = np.zeros(nSamples)  # Y position from calculations
        Py_Solv = np.zeros(nSamples)  # Y position from RK4 solver
        Lv_Fbck = np.zeros(nSamples)  # Linear vel from feedback
        Lv_Calc = np.zeros(nSamples)  # Linear vel from calculations
        Lv_Solv = np.zeros(nSamples)  # Linear vel from RK4 solver
        Av_Fbck = np.zeros(nSamples)  # Angular vel from feedback
        Av_Calc = np.zeros(nSamples)  # Angular vel from calculations
        Av_Solv = np.zeros(nSamples)  # Angular vel from RK4 solver
        Wd_Fbck = np.zeros(nSamples)  # Angular accel from feedback
        Wd_Calc = np.zeros(nSamples)  # Angular accel from calculations
        Wd_Solv = np.zeros(nSamples)  # Angular accel from RK4 solver
        Tb_Fbck = np.zeros(nSamples)  # Body torque from feedback
        Tb_Calc = np.zeros(nSamples)  # Body torque from calculations
        Tb_Solv = np.zeros(nSamples)  # Body torque from RK4 solver
        TE_Fbck = np.zeros(nSamples)  # Total energy from feedback
        TE_Solv = np.zeros(nSamples)  # Total energy from RK4 solver
    
    # Initialize values for state variables array.

    Pb   = body.getPosition()
    Vb   = body.getLinearVel()
    a    = jh.getAngle()
    adot = jh.getAngleRate()

    S[0] = 0.0
    S[1] = Pb[0]
    S[2] = Pb[1]
    S[3] = Vb[0]
    S[4] = Vb[1]
    S[5] = THETA_0 + a
    S[6] = adot
    
    # Instantiate Runge-Kutta 4th order ode solver and initialize.
    
    rk4 = RK4_Solver(T_STEP,nSvar)
    
    rk4.init(S)
    
    # Take a very small step in ODE to set rates and feedback data.
    
    #world.step(T_STEP/1000.0)
        
    # Loop until termination event or simulation stop condition reached.
    
    loopFlag = True
    t        = S[0]
    i        = 0
    while loopFlag and i < nSamples:
        
        # Check for terminate event.
        for e in pygame.event.get():
            if e.type==QUIT:
                loopFlag=False
            if e.type==KEYDOWN:
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
            
        # Get current body state and state derivatives from RK4 solver.
        dS  = dotS(nSvar,S)
        Px  = (S[1], S[2], 0.0)    # position
        Vx  = (dS[1], dS[2], 0.0)  # linear velocity
        Ax  = (dS[3], dS[4], 0.0)  # linear acceleration
        Tx  = S[5]                 # rotation angle
        Wx  = (0.0, 0.0, S[6])     # angular velocity
        Wdx = (0.0, 0.0, dS[6])    # angular acceleration
        
        # Get current ODE hinge joint and body world space positions.
        Pj = jh.getAnchor()
        Pb = body.getPosition()

        # Draw the body and lines representing the joint.
        pygame.draw.line(screen, J_COLOR, coord(Pj), coord(Pb), J_LWID)
        pygame.draw.circle(screen, B_COLOR, coord(Pb, True), B_CRAD, 0)
        pygame.draw.line(screen, X_COLOR, coord(Pj), coord(Px), 1)
        pygame.draw.circle(screen, X_COLOR, coord(Px, True), B_CRAD, 1)
        
        # Display updated screen.
        pygame.display.flip()
        if SAVE_ANIM:
            istr = format_string("%04d", i)
            fpth = "./anim/tutorial2bob_" + istr + ".png"
            pygame.image.save(screen, fpth)
            
        # Evaluate body dynamics...
        #... get hinge joint angle and angular rate.
        a     = jh.getAngle()
        adot  = jh.getAngleRate()
        theta = THETA_0 + a  # joint angle wrt to reference -Y axis
        #... get body rotation angle about its axis.
        Qref  = body.getQuaternion()
        U     = (Qref[1], Qref[2], Qref[3])
        normU = vecMag(U)
        if normU > 0.0: vdir = vecDotP(vecDivS(U,normU),B_AXIS)
        else:           vdir = -1.0 
        alpha = vdir*2*atan2(normU,Qref[0])
        #... calculate distance between joint and body.
        Pdst = vecSub(Pb,Pj)
        Pdir = unitVec(Pdst)
        #... get joint position wrt to Com in body frame.
        Jdst = vecRotZ(vecSub(Pj,Pb), theta)
        Jdir = unitVec(Jdst)
        psi  = acosUVecDotP(Jdir,B_NORM)
        if PRINT_DATA:
            print("t = %8.4f" % (t) )
            print("theta = %8.4f  a = %8.4f  adot = %8.4f" % \
                (theta*DPR, a*DPR, adot) )
            print("alpha = %8.4f  psi  = %8.4f" % \
                (alpha*DPR, psi*DPR) )
            printVec('Pj',Pj)
            printVec('Pb',Pb)
            printVec('Pdst',Pdst)
            printVec('Jdst',Jdst)
        #... get body angular and linear velocities.
        Wref = body.getAngularVel()
        Vref = body.getLinearVel()
        Vact = vecCrossP(Wref,Pdst)
        Verr = vecSub(Vref,Vact)
        if PRINT_DATA:
            printVec('Wref',Wref)
            printVec('Vref',Vref)
            printVec('Vact',Vact)
            printVec('Verr',Verr)
        #... rotate velocity into body and tangential/normal axis
        #    frame (both should be identically aligned).
        Vb = vecRotZ(Vact,theta)
        Vj = vecRotZ(Vb,psi)
        if PRINT_DATA:
            printVec('Vb',Vb)
            printVec('Vj',Vj)
        #... rotate gravity acceleration and force vectors into body
        #    and body tangential/normal axis (joint-aligned) frames.
        Ag   = GRAVITY
        Agb  = vecRotZ(Ag,theta)
        Agj  = vecRotZ(Agb,psi)
        Agjt = rejectionVecAfromUB(Agj,B_NORM)
        Agjn = projectionVecAonUB(Agj,B_NORM)
        Fg   = vecMulS(Ag,B_MASS)
        Fgb  = vecMulS(Agb,B_MASS)
        Fgj  = vecMulS(Agj,B_MASS)
        Fgjt = rejectionVecAfromUB(Fgj,B_NORM)
        Fgjn = projectionVecAonUB(Fgj,B_NORM)
        if PRINT_DATA:
            printVec('Fgb',Fgb)
            printVec('Fgj',Fgj)
        #... get moment of inertia at joint for current body rotation.
        '''
        R     = rotToM(body.getRotation())
        Irot  = matMulM(R,matMulM(B_Iinv,transposeM(R)))
        Idotv = matMulV(vecToSkewMat(Vact),matMulV(Irot,Vact))
        Idotw = matMulV(vecToSkewMat(Wref),matMulV(Irot,Wref))
        print t, theta*DPR, matMulV(R,B_NORM)
        print Irot
        print Idotv
        print Idotw
        '''
        Izzj = B_Izzc + B_MASS*vecMagSq(Jdst)                
        #... get joint force and torque on body; rotate into body frame.
        if i == 0:
            #___ obtain from calculated joint forces and torque using 
            #    initial conditions (v=0, w=0).
            Wdot = vecDivS(vecCrossP(Fgjt,Jdst),Izzj)
            Awj  = vecAdd(vecCrossP(Jdst,Wdot),vecCrossP(Wref,Vj))
            Awn  = projectionVecAonUB(Awj,B_NORM)
            Awt  = rejectionVecAfromUB(Awj,B_NORM)
            Fwn  = vecMulS(Awn,B_MASS)
            Fwt  = vecMulS(Awt,Izzj/vecMagSq(Jdst))
            Fbj  = vecAdd(Fwt,Fwn)
            Fj   = vecSub(vecRotZ(Fbj,-(theta+psi)),Fg)
            Tqj  = vecMulS(Wdot,B_Izzc)
        else:
            #___ obtain from joint feedback force and torque (note: fb[0]
            #    and fb[1] are the force and torque generated from body 2
            #    and propagated by the joint to anchor point in body 1).
            fb  = jh.getFeedback()
            Fj  = fb[0]  # force at body 2 CoM
            Tqj = fb[1]  # torque at body 2 CoM
            if PRINT_DATA:
                printVec('fb[0]',fb[0])
                printVec('fb[1]',fb[1])
        if PRINT_DATA:
            printVec('Fj',Fj)
            printVec('Tqj',Tqj)
        #... calculate tangential and normal forces in joint-aligned frame.
        Fb   = vecAdd(Fj,Fg)      # remove anti-gravity constraint on joint
        Fbb  = vecRotZ(Fb,theta)  # resultant force in body frame
        Fbj  = vecRotZ(Fbb,psi)   # ... in body joint-aligned frame
        Fbjt = rejectionVecAfromUB(Fbj,B_NORM) # use only Fbj +x component
        Fbjn = projectionVecAonUB(Fbj,B_NORM)  # use only Fbj +y component
        #... calculate total tangential and normal accels using feedback.
        Abjt = vecAdd(vecDivS(Fbjt,Izzj/vecMagSq(Jdst)),Agjt)
        Abjn = vecAdd(vecDivS(Fbjn,B_MASS),Agjn)
        #... calculate angular acceleration from Euler's equations.
        Wdot = vecDivS(vecCrossP(Fgjt,Jdst),Izzj)
        #... calculate normal and tangential accelerations from Newton's
        #    law of motion (note: Jdst X Wdot yields vectors in the
        #    direction of the body frame +x axis, while Wref x Vj yields
        #    vectors in the direction of the body frame +y axis).
        Awj = vecAdd(vecCrossP(Jdst,Wdot),vecCrossP(Wref,Vj))
        Awn = projectionVecAonUB(Awj,B_NORM)
        Awt = rejectionVecAfromUB(Awj,B_NORM)
        if PRINT_DATA:
            printVec('Awt',Awt)
            printVec('Awn',Awn)
        #... calculate forces and torque at joint to compare with feedback.
        Fwn = vecMulS(Awn,B_MASS)
        Fwt = vecMulS(Awt,Izzj/vecMagSq(Jdst))
        Fcj = vecAdd(Fwt,Fwn)
        Fc  = vecSub(vecRotZ(Fcj,-(theta+psi)),Fg) # include anti-gravity constraint
        Tqc = vecMulS(Wdot,B_Izzc)
        #... compute applied forces and torque from RK4 solver.
        Axj = vecRotZ(Ax,Tx+psi)  # total acceleration in body frame
        Axn = vecSub(projectionVecAonUB(Axj,B_NORM),Agjn)  # normal-g component
        Axt = vecSub(rejectionVecAfromUB(Axj,B_NORM),Agjt) # tangent-g component
        Fxn = vecMulS(Axn,B_MASS)                   # pure translational forces
        Fxt = vecMulS(Axt,B_Izzj/(LENGTH**2))       # moment inducing forces
        Fxj = vecAdd(Fxt,Fxn)                       # total in body frame
        Fx  = vecSub(vecRotZ(Fxj,-(theta+psi)),Fg)  # include anti-gravity constraint
        Tqx = vecMulS(Wdx,B_Izzc)                   # torque on body
        '''
        #... move torque from body CoM to joint.
        Tqxj = vecMulS(Wdx,B_Izzj)
        Tqxb = vecMulS(Wdx,B_Izzc)
        Fdel = vecMulS(vecCrossP(Pdst,vecSub(Tqxj,Tqxb)),vecMag(Pdst))
        Tdel = vecCrossP(vecMulS(Pdst,-1),Fdel)
        print vecMag(Fdel), (Tqxj[2]-Tqxb[2])/LENGTH
        print Tqxj[2], Tqxb[2], Tqj[2], vecAdd(Tqj,Tdel)[2]
        '''
        if PRINT_DATA:
            printVec('Fj',Fj)
            printVec('Fc',Fc)
            printVec('FX',Fx)
            printVec('Fbj',Fbj)
            printVec('Fcj',Fcj)
            printVec('Fxj',Fxj)
            printVec('Tqj',Tqj)
            printVec('Tqc',Tqc)
            printVec('Fqx',Tqx)
        #... compute total energy of system.
        ''' 
        Note: Sum of kinetic energy due to mass's CoM velocity along
              pendulum arc and mass's rotation rate about its CoM can
              be replaced with rotational kinetic enegy of mass's CoM 
              about the pendulum pivot joint (see eq 2.63 on pg 81 and
              eqs 2.71 and 2.72 on pg 83 of REF [2]).
        totE_ode = 0.5*B_MASS*vecMagSq(Vref) + \
                   0.5*B_Izzc*vecMagSq(Wref) + B_MASS*G*Pb[1] - MIN_PE
        totE_rk4 = 0.5*B_MASS*vecMagSq(Vx) + \
                   0.5*B_Izzc*vecMagSq(Wx) + B_MASS*G*Px[1] - MIN_PE
        '''
        totE_ode = 0.5*B_Izzj*adot**2 + B_MASS*G*LENGTH*(1.0 - cos(theta))
        totE_rk4 = 0.5*B_Izzj*vecMagSq(Wx) + B_MASS*G*LENGTH*(1.0 - cos(Tx))
        
        # Collect data for plotting.
        if PLOT_DATA:
            Time[i]    = t

            Ft_Fbck[i] = Fj[0]/G
            Ft_Calc[i] = Fc[0]/G
            Ft_Solv[i] = Fx[0]/G
            
            Fn_Fbck[i] = Fj[1]/G 
            Fn_Calc[i] = Fc[1]/G
            Fn_Solv[i] = Fx[1]/G
            
            Tb_Fbck[i] = Tqj[2]
            Tb_Calc[i] = Tqc[2]
            Tb_Solv[i] = Tqx[2]
      
            At_Fbck[i] = Abjt[0]/G
            At_Calc[i] = (Awt[0] + Agjt[0])/G
            At_Solv[i] = Axj[0]/G
            
            An_Fbck[i] = Abjn[1]/G
            An_Calc[i] = (Awn[1] + Agjn[1])/G
            An_Solv[i] = Axj[1]/G
            
            Py_Fbck[i] = Pb[1]
            Py_Calc[i] = Pdst[1]
            Py_Solv[i] = Px[1]
            
            Lv_Fbck[i] = vecMag(Vref)
            Lv_Calc[i] = vecMag(Vact)
            Lv_Solv[i] = vecMag(Vx)
            
            Av_Fbck[i] = adot*DPR
            Av_Calc[i] = Wref[2]*DPR
            Av_Solv[i] = Wx[2]*DPR
            
            Wd_Fbck[i] = Tqj[2]/B_Izzc
            Wd_Calc[i] = Wdot[2]
            Wd_Solv[i] = Wdx[2]
            
            TE_Fbck[i] = totE_ode
            TE_Solv[i] = totE_rk4

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
            t = t + N_TIME
            i = i + 1
    
        # Create and show the plots.
    
        figures = []
        
        figures.append(plt.figure(1, figsize=(8,6), dpi=80))
        plt.title("Horizontal Force (g's) on Joint")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Ft_Fbck,'b-',Time,Ft_Calc,'r:2',\
                    Time,Ft_Solv,'g:3')           
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(2, figsize=(8,6), dpi=80))
        plt.title("Vertical Force (g's) on Joint")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Fn_Fbck,'b-',Time,Fn_Calc,'r:2',\
                    Time,Fn_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper center')
        
        figures.append(plt.figure(3, figsize=(8,6), dpi=80))
        plt.title("Torque on Body")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Tb_Fbck,'b-',Time,Tb_Calc,'r:2',\
                    Time,Tb_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(4, figsize=(8,6), dpi=80))
        plt.title("Total Tangential Acceleration (g's) on Body")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,At_Fbck,'b-',Time,At_Calc,'r:2',\
                    Time,At_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(5, figsize=(8,6), dpi=80))
        plt.title("Total Normal Acceleration (g's) on Body")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,An_Fbck,'b-',Time,An_Calc,'r:2',\
                    Time,An_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper center')
        
        figures.append(plt.figure(6, figsize=(8,6), dpi=80))
        plt.title("Body Linear Velocity (m/sec)")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Lv_Fbck,'b-',Time,Lv_Calc,'r:2', 
                 Time,Lv_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(7, figsize=(8,6), dpi=80))
        plt.title("Body Angular Velocity (deg/sec)")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Av_Fbck,'b-',Time,Av_Calc,'r:2',\
                    Time,Av_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(8, figsize=(8,6), dpi=80))
        plt.title("Body Angular Acceleration (rad/sec/sec)")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Wd_Fbck,'b-',Time,Wd_Calc,'r:2',\
                    Time,Wd_Solv,'g:3')
        plt.legend(('ODEsolver','Computed','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(9, figsize=(8,6), dpi=80))
        plt.title("Total System Energy (wrt pendulum at rest)")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,TE_Fbck,'b-',Time,TE_Solv,'g:3')
        plt.legend(('ODEsolver','RK4solver'),loc='upper left')
        
        figures.append(plt.figure(10, figsize=(8,6), dpi=80))
        plt.title("Body Kinematic Errors (RK4 - ODE)")
        plt.xlabel('Time (sec)')
        plt.xlim(0.0,T_STOP)
        plt.grid()
        plt.plot(Time,Py_Solv-Py_Fbck,'r-', Time,Lv_Solv-Lv_Fbck,'g-',Time,Av_Solv-Av_Fbck,'b-')
        plt.legend(('Y pos. (m)', 'Lin. vel. (m/sec)','Ang. vel. (deg/sec)'),loc='upper left')
        
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
                