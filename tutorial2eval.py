# -*- coding: utf-8 -*-
# pylint: disable=trailing-whitespace,bad-whitespace,invalid-name,anomalous-backslash-in-string

""" 
Tutorial2x procedures for dynamics evaluation of Compound Double Pendulum (CDP)
system modeled with Python Open Dynamics Engine (PyODE):

    1) compute_wdot_vdot(j1, theta0_1, j2, theta0_2, g)
    2) calcFwdDynamicsODE(jList, Theta0_1, Theta0_2, g, PRINT_EVAL)
  
"""

# File: tutorial2eval.py
# Auth: Gary E. Deschaines
# Date: 29 Jun 2020
# Prog: Tutorial2x procedures for dynamics evaluation of CDP system ODE model. 
# Desc: Procedures to compute joint angular accelerations and connected body 
#       linear accelerations for compound double pendulum (CDP) system modeled
#       in ODE.
#
# References (as indexed in other tutorial2x Python scripts):
#
# [3] Neumann, Erik, "Double Pendulum," Eric Neumann's myPhysicsLab.com,
#     web site, Dec. 18, 2010. Web available at
#     https://www.myphysicslab.com/pendulum/double-pendulum-en.html
#
# [5] Lynch, Kevin M. and Park, Frank C., "Modern Robotics:  Mechanics, 
#     Planning, and Control," 3rd printing 2019, Cambridge University
#     Press, 2017. Web available at
#     http://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf
#
# [6] John J. Craig, "Introduction to Robotics: Mechanics and Control",
#     3rd ed., Pearson Prentice Hall, Pearson Education, Inc., Upper 
#     Saddle River, NJ, 2005
#
# [7] Liu, Karen and Jain, Sumit, "A Quick Tutorial on Multibody Dynamics,"
#     Tech Report GIT-GVU-15-01-1, School of Interactive Computing, Georgia
#     Institute of Technology, 2012. Web available at
#     https://studylib.net/doc/14301909/a-quick-tutorial-on-multibody-dynamics
#
# [8] Dr. Robert L. William II, "Robot Mechanics: Notesbook Supplement
#     for ME 4290/5290 Mechanics and Control of Robotic Manipulators",
#     Ohio University, Mech. Engineering, Spring 2015; web available at
#     https://people.ohio.edu/williams/html/PDF/Supplement4290.pdf
#
# Disclaimer:
#
# See DISCLAIMER

def compute_total_torque(j1, j2):
    """
    Computes and returns total torque for given ODE joints from 
    associated joint feedback force and torque data.
    """
    from numpy import zeros
    from vecMath import vecSub, vecCrossP
    
    # Get current ODE joint and body world space positions.
    j1Pos = j1.getAnchor()
    j2Pos = j2.getAnchor()
    b1    = j1.getBody(0)
    b2    = j2.getBody(0)
    b1Pos = b1.getPosition()
    b2Pos = b2.getPosition()
    
    # Get joint feedback strutures holding forces and torques on connected
    # bodies. These forces and torques are applied at the connected body's
    # center of mass (COM) and are represented as vectors in ODE inertial
    # world reference frame. 
    fb1 = j1.getFeedback()
    fb2 = j2.getFeedback()
    
    # Compute total torque on rod-body assembly (i.e., rigid body link) COM
    # from torques and forces of parent and child links at their respective
    # connecting joint locations (see sec 6.2 on pg 19 and eq 73 on pg 21
    # of ref [7]).
    tau = zeros((2,1))
    C1  = vecSub(b1Pos,j1Pos)
    D2  = vecSub(j2Pos,j1Pos)
    tau[0] = fb1[1][2] \
           + vecCrossP(C1,fb1[0])[2] \
           + fb2[3][2] \
           + vecCrossP(vecSub(D2,C1),fb2[2])[2]
    C2  = vecSub(b2Pos,j2Pos)
    tau[1] = fb2[1][2] \
           + vecCrossP(C2,fb2[0])[2]
    return tau
    

def compute_joint_torque(j1, j2):
    """
    Computes and returns joint torque for given ODE joints from 
    associated joint feedback force and torque data.
    """
    from numpy import zeros
    from vecMath import vecSub, vecCrossP
    
    # Get current ODE joint and body world space positions.
    j1Pos = j1.getAnchor()
    j2Pos = j2.getAnchor()
    b1    = j1.getBody(0)
    b2    = j2.getBody(0)
    b1Pos = b1.getPosition()
    b2Pos = b2.getPosition()
    
    # Get joint feedback strutures holding forces and torques on connected
    # bodies. These forces and torques are applied at the connected body's
    # center of mass (COM) and are represented as vectors in ODE inertial
    # world reference frame. 
    fb1 = j1.getFeedback()
    fb2 = j2.getFeedback()
    
    # Compute total torque on rod-body assembly (i.e., rigid body link) COM
    # from torques and forces of parent and child links at their respective
    # connecting joint locations (see eqs 6.51, 6.52 and 6.53 in sec 6.5 of
    # ref [6]).
    tau = zeros((2,1))
    C2  = vecSub(b2Pos,j2Pos)
    tau[1] = fb2[1][2] \
           + vecCrossP(C2,fb2[0])[2]
    C1  = vecSub(b1Pos,j1Pos)
    D2  = vecSub(j2Pos,j1Pos)
    tau[0] = fb1[1][2] \
           + vecCrossP(C1,fb1[0])[2] \
           + tau[1] \
           + vecCrossP(D2,fb2[0])[2]
    return tau


def compute_wdot_vdot(j1, theta0_1, j2, theta0_2, g):
    """
    Computes angular accelerations of the given ODE joints and 
    linear accelerations of attached rigid bodies for a double
    pendulum system with its motion constrained to an XY plane
    as depicted in the diagram below.
    
    The angular accelerations are computed by application of 
    state-space rigid body dynamics equations in the form:
        
      qdd = inv(M(q))*(tau - C(q,qd) - G(q))  
        
    where in this case, the generalized coordinates q are theta
    for joints angles, qd are w for joint angle rates and wdot 
    is qdd. Computed values for wdot are then used to compute 
    linear accelarations vdot as XYZ space vectors for bodies 
    b0, b1 and b2 (refer to pp. 269-275 of ref [5]).

          +Yo
           |
           |
        j1-o-b0->+Xo  |wb = b0AVel,  vb = b0LVel
           :\         |@b = wdot[0], ab = vdot[0]
           : \
           :  \  +Ybi |T1 = T1o + j1angle
           :   \  :   |w1 = j1omega
           :=T1>\ :   |@1 = wdot[1]   
                 \:                         r = b1Pos - j1Pos
               j2-o-b1->+Xbi |wb = b1AVel,  vb = b1LVel
                  :\         |@b = wdot[1], ab = vdot[1]
                  : \
                  :  \  +Ybi |T2 = T2o + j1angle + j2angle
                  :   \  :   |w2 = j1omega + j2omega
                  :=T2>\ :   |@2 = wdot[2]
                        \:                         r = b2Pos - j2Pos
                         o-b2->+Xbi |wb = b2AVel,  vb = b2LVel
                                    |@b = wdot[2], ab = vdot[2]
    
    Note: The initial orientation of body frame +Xbi and +Ybi axes
          are shown for T1o, T2o, j1angle and j2angle equal to zero.
          During pendulum motion the instantaneous orientation has
          +Yb axis collinear with the rod from body to joint. This
          corresponds to joint rotation angles j1angle and j2angle,
          and joint rotation rates j1omega and j2omega.
          
    """
    from numpy import zeros, dot
    from scipy.linalg import inv
    from math import cos, sin
    from vecMath import vecMag, vecSub, vecDotP
    
    # Get current ODE joint and body world space positions.
    j1Pos = j1.getAnchor()
    j2Pos = j2.getAnchor()
    b1    = j1.getBody(0)
    b2    = j2.getBody(0)
    b1Pos = b1.getPosition()
    b2Pos = b2.getPosition()
    
    # Get body angular velocities (inertial omega w).
    b1AVel = b1.getAngularVel()
    b2AVel = b2.getAngularVel()
        
    # Get joint angle and angular rates (for generalized coordinates q, and qd).
    j1angle = j1.getAngle()
    j1omega = j1.getAngleRate()
    j2angle = j2.getAngle()
    j2omega = j2.getAngleRate()
    
    # Convert joint angle and rates to absolute (refer to diagram notes).
    j1theta = theta0_1 + j1angle
    j2theta = theta0_2 + j1angle + j2angle
    j1thdot = j1omega
    j2thdot = j2omega + j1omega    
    
    # Acquire EQOM parameters for current pendulum geometry.
    L1   = vecMag(vecSub(b1Pos,j1Pos))
    L2   = vecMag(vecSub(b2Pos,j2Pos))
    L1sq = L1*L1
    L2sq = L2*L2
    L12  = L1*L2
    c1   = cos(j1theta)
    s1   = sin(j1theta)
    c2   = cos(j2theta)
    s2   = sin(j2theta)
    c1m2 = cos(j1theta - j2theta)
    s1m2 = sin(j1theta - j2theta)
    
    #
    # Note: The following expressions for mass, Coriolis and centripetal
    #       torques, and gravitational torque matrices were derived with
    #       the Octave script diffeqs_cdp.m provided in this Tutorial2x
    #       collection.
    #
    
    # Acquire EQOM parameters for body mass properties.
    mass1 = b1.getMass()
    mass2 = b2.getMass()
    m1    = mass1.mass
    m2    = mass2.mass
    I1c   = mass1.I[2][2]
    I2c   = mass2.I[2][2]
    I1    = I1c + m1*L1sq
    I2    = I2c
    
    # Compute mass matrix.
    M = zeros((2,2))
    M[0,0] = I1 + m1*L1sq
    M[0,1] = m2*L1*L2*c1m2
    M[1,0] = M[0,1]
    M[1,1] = I2 + m2*L2sq
    
    # Compute Coriolis and centripetal torques matrix.
    C = zeros((2,1))
    C[0,0] = m2*L12*s1m2*j2thdot**2
    C[1,0] = -m2*L12*s1m2*j1thdot**2
    
    # Compute gravitational torque matrix.
    G = zeros((2,1))
    G[0,0] = (m1 + m2)*g*L1*s1
    G[1,0] = m2*g*L2*s2
    
    # Compute total torque of each joint for the rod-body assembly. 
    tau = compute_total_torque(j1, j2)
    
    # Apply dynamics equations.
    wdot = zeros((3,1))
    wdot[1:3,0] = dot(inv(M),(tau - C - G))[0:2,0]
    
    # Compute body inertial acceleration components (see eqs 1-4 in ref [3]).
    vdot = zeros((3,3))
    dth1  = vecDotP(b1AVel,j1.getAxis())
    assert (dth1 - j1thdot) < 1.0e-6
    dth2  = vecDotP(b2AVel,j2.getAxis())
    assert (dth2 - j2thdot) < 1.0e-6
    ddth1 = wdot[1]
    ddth2 = wdot[2]
    vdot[1,0] = L1*(ddth1*c1 - dth1**2*s1)
    vdot[1,1] = L1*(ddth1*s1 + dth1**2*c1)
    vdot[1,2] = 0.0
    vdot[2,0] = vdot[1,0] + L2*(ddth2*c2 - dth2**2*s2)
    vdot[2,1] = vdot[1,1] + L2*(ddth2*s2 + dth2**2*c2)
    vdot[2,2] = 0.0

    return (wdot, vdot)


def calcFwdDynamicsODE(jList, theta0_1, theta0_2, g, print_eval):
    """
    Calculates forward dynamics using Articulated-Body Algorithm for ODE
    model of compound double pendulum system as depicted in the diagram
    presented within compute_wdot_vdot() procedure above. Uses forces
    and torques from joint feedback data to compute linear and angular
    accelerations which are not provided by ODE.
    """
    from scipy.linalg import pinv
    from tutorial2util import printVec
    from vecMath import vecAdd, vecSub, vecMulS, vecDivS, vecCrossP
    from vecMath import vecToSkewMat, transposeM, matDotV

    th0   = [0.0, theta0_1, theta0_2]  # initial joint angles
    Ag    = (0.0, g, 0.0)              # gravitational acceleration offset
    Z_VEC = (0.0, 0.0, 0.0)            # zero vector
    
    if print_eval:
        print('calcFwdDynamicsODE:')
        
    # Recursion work space variables (measures in global reference frame).
    # Note: The following variable lists contain a zero scalar or vector 
    #       (Z_VEC) in the 0th entry to simplify outbound and inbound 
    #       recursion computations below.
    rth = [0.0,0.0,0.0,0.0]    # joint relative rotation angles
    ath = [0.0,0.0,0.0,0.0]    # joint absolute rotation angles
    Pj  = [Z_VEC,Z_VEC,Z_VEC]  # joint position vectors
    Vj  = [Z_VEC,Z_VEC,Z_VEC]  # joint linear velocity vectors
    Aj  = [Z_VEC,Z_VEC,Z_VEC]  # joint linear acceleration vectors
    w   = [Z_VEC,Z_VEC,Z_VEC]  # joint/body angular velocity vectors
    a   = [Z_VEC,Z_VEC,Z_VEC]  # joint/body angular acceleration vectors
    Pb  = [Z_VEC,Z_VEC,Z_VEC]  # body position vectors
    Vb  = [Z_VEC,Z_VEC,Z_VEC]  # body linear velocity vectors
    Ab  = [Z_VEC,Z_VEC,Z_VEC]  # body linear acceleration vectors
    Wb  = [Z_VEC,Z_VEC,Z_VEC]  # body angular velocity vectors
    Fb  = [Z_VEC,Z_VEC,Z_VEC]  # inertial forces (in joint/body frame)
    Nb  = [Z_VEC,Z_VEC,Z_VEC]  # inertial moments (in joint/body frame)
    fb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # internal forces (in joint/body frame)
    nb  = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # total torque at inboard joint
    nbo = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # moment about body cg from force at outboard joint
    nbi = [Z_VEC,Z_VEC,Z_VEC,Z_VEC]  # moment about inboard joint from force at body cg
    tau = [0.0,0.0,0.0]        # total torque on inboard joint
    
    # Outbound kinematics iteration (see sec 11.2 of ref[8], and sec 7.2 with 
    # eqs 76 & 79 of ref [7]).
    if print_eval:
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
        # ... joint inertial linear velocity (eq 76 of ref[7])
        Vj[ij] = vecAdd(Vj[ij-1],vecCrossP(w[ij-1],D0))
        # ... joint inertial linear acceleration (eq 79 of ref[7])
        Aj[ij] = vecAdd(Aj[ij-1],\
                    vecAdd(vecCrossP(a[ij-1],D0),\
                        vecCrossP(w[ij-1],vecCrossP(w[ij-1],D0))))
        # ... joint inertial angular acceleration (from joint feedback) by
        #     solving Ftot = m*(a[ij]xC1 + w[ij]x(w[ij]xC1) + Aj[ij]) for 
        #     a[ij]
        fbck1 = j.getFeedback()  # for this joint's force on body
        if ij < kj:
            # account for next joint's force on body
            fbck2 = jList[ij+1].getFeedback()
            fbtot = vecAdd(fbck1[0],fbck2[2])
        else:
            fbtot = fbck1[0]
        # note: remove gravitational offset applied by ODE    
        a[ij] = vecSub(vecSub(vecDivS(fbtot,m),Ag),\
                       vecAdd(vecCrossP(w[ij],vecCrossP(w[ij],C1)),Aj[ij]))
        a[ij] = matDotV(pinv(transposeM(vecToSkewMat(C1))),a[ij])
        # ... body inertial linear and angular velocity
        Vb[ij] = b.getLinearVel()
        Wb[ij] = b.getAngularVel()
        # ... body inertial linear acceleration (eq 79 of ref[7])
        Ab[ij] = vecAdd(Aj[ij],\
                    vecAdd(vecCrossP(a[ij],C1),\
                        vecCrossP(w[ij],vecCrossP(w[ij],C1))))
        # ... inertial loading (kinetics)
        Fb[ij] = vecMulS(vecAdd(Ab[ij],Ag),m)  # reapply gravitational offset
        Nb[ij] = vecAdd(matDotV(I,a[ij]),\
                        vecCrossP(w[ij],matDotV(I,w[ij])))
        if print_eval:
            printVec('j'+str(ij), 'w', w[ij])
            printVec('j'+str(ij), 'a', a[ij])
            printVec('j'+str(ij), 'Vj', Vj[ij])
            printVec('j'+str(ij), 'Aj', Aj[ij])
            printVec('b'+str(ij), 'Vb', Vb[ij])
            printVec('b'+str(ij), 'Wb', Wb[ij])
            printVec('b'+str(ij), 'Ab', Ab[ij])
            printVec('b'+str(ij), 'Fb', Fb[ij])
            printVec('b'+str(ij), 'Nb', Nb[ij])       
        ij = ij + 1
        
    # Inbound articulated body inertia and bias force recursion (see
    # sec 7.3 and eqs 72 & 73 of ref [7]).
    if print_eval:
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
        if print_eval:
            printVec('j'+str(ij), 'fb', fb[ij])
            printVec('j'+str(ij), 'nb', nb[ij])
            printVec('j'+str(ij), 'nbi', nbi[ij])
            printVec('j'+str(ij), 'nbo', nbo[ij])
        ij = ij - 1
    
    if print_eval:
        print('...return.')
        
    return (Pj,rth,ath,w,a,Pb,Vb,Wb,Ab,(tau[1],tau[2]))
