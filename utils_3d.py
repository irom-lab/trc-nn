import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


##########################################################################
##### Conversion between S03 and euler angle, quaternion, axis-angle #####
##########################################################################

# Euler angle (? convention) to SO(3), multiply 3 matrix
# SO(3) to Euler angle

# Convert Euler rotation to SO(3), i=1/2/3=rpy
def euler2rot(i, z):
    out = np.eye(3)
    s = np.sin(z)
    c = np.cos(z)

    if i == 1:  # roll
        out[1,1] = c
        out[1,2] = -s
        out[2,1] = s
        out[2,2] = c
    elif i == 2: # pitch
        out[0,0] = c
        out[0,2] = s
        out[2,0] = -s
        out[2,2] = c
    elif i == 3: # yaw
        out[0,0] = c
        out[0,1] = -s
        out[1,0] = s
        out[1,1] = c
  
    return out

# Convert quatenornion (a,b,c,w, unnormalized) to SO(3)
def quat2rot(z, w_first=False):
    if w_first:
        a = z[1]
        b = z[2]
        c = z[3]
        w = z[0]
    else:
        a = z[0]
        b = z[1]
        c = z[2]
        w = z[3]

    out = np.zeros((3,3))
    out[0,0] = w**2+a**2-b**2-c**2
    out[0,1] = 2*a*b-2*c*w
    out[0,2] = 2*a*c+2*b*w
    out[1,0] = 2*a*b+2*c*w
    out[1,1] = w**2-a**2+b**2-c**2
    out[1,2] = 2*b*c-2*a*w
    out[2,0] = 2*a*c-2*b*w
    out[2,1] = 2*b*c+2*a*w
    out[2,2] = w**2-a**2-b**2+c**2

    return out

# Convert SO(3) to quatenornion (a,b,c,w, unnormalized)
# https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_quaternion
def rot2quat(r):
    w = 0.5*np.sqrt(1+np.trace(r))
    a = 0.25/w*(r[2,1]-r[1,2])
    b = 0.25/w*(r[0,2]-r[2,0])
    c = 0.25/w*(r[1,0]-r[0,1])
    return np.array([a,b,c,w])

# Axis-angle to S03
# SO3 to axis-angle

############################################################################
#################### Conversion with normal vector #########################
############################################################################

# Find SO(3) that rotate x vector to y, not unique since not aligning frames but just normals (x,y required to be unit vectors!!!!)
def vec2rot(x,y):
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# rotate around cross product of x and y by arccos(dot(x,y))
# wont work if exactly opposite direction for x and y
    v = np.cross(x,y)
    s = np.linalg.norm(v)
    c = np.dot(x,y)
    vs = skew3D(v)

    # if x == -y

    return np.eye(3) + vs + vs.dot(vs)*(1/(1+c))

# Rotate a vector v by a quaternion q (a,b,c,w), return a 3D vector
def vecQuat2vec(v,q):
    r = np.concatenate((v,[0]))  # add zero to the end of the array
    q_conj = np.array([-q[0],-q[1],-q[2],q[3]])
    out = quatMult(quatMult(np.array(q),r),q_conj)[:3]
    return out/np.linalg.norm(out)

# Find quaternion that rotates x vector to y , not unique since not aligning frames but just normals
# https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
def vecs2quat(x,y):
    out = np.zeros(4)
    out[:3] = np.cross(x, y)
    out[3] = np.linalg.norm(x)*np.linalg.norm(y)+np.dot(x, y)
    if np.linalg.norm(out) < 1e-4:
        return np.append(-x, [0])  # 180 rotation
    return out/np.linalg.norm(out)

########################################################################################################

# Multiply two quaternions (a,b,c,w)
def quatMult(p, q):
    w = p[3]*q[3] - np.dot(p[:3], q[:3])
    abc = p[3]*q[:3] + q[3]*p[:3] + np.cross(p[:3], q[:3])
    return np.hstack((abc, w))

# Convert 3D vector to 3x3 skew-symmetric matrix
def skew3D(z):
    return np.array([[0,    -z[2], z[1]],
                    [z[2],  0,    -z[0]],
                    [-z[1], z[0], 0]])

# Get angle between two vectors
def angleBwVec(p,q):
    p = np.array(p)
    q = np.array(q)
    ct = np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))
    return np.arccos(ct)

# Dot product of two lists
# def dot_list(K, L):
#    if len(K) != len(L):
#       return 0
#    return sum(i[0] * i[1] for i in zip(K, L))

# def norm_list(K):
#    return np.sqrt(sum(i**2 for i in K))

def SO3_6D_np(b1, a2):
    b2 = a2 - np.dot(b1, a2)*b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return b2, b3
