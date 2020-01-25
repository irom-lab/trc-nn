import math
import time
import numpy as np
import ray
import sys
import pybullet as p

def getParameters():
    params = {}

    params['imgW'] = 480
    params['imgH'] = 480

    params['imgW_orig'] = 1024
    params['imgH_orig'] = 768

    # p.resetDebugVisualizerCamera(0.40, 225, -60, [0.50, 0.0, 0.04])  # 40cm away
    # params['viewMatPanda'] = [-0.70710688829422, 0.6123723387718201, -0.3535534143447876, 0.0, -0.7071066498756409, -0.6123725175857544, 0.35355353355407715, 0.0, 0.0, 0.5000001192092896, 0.8660253882408142, 0.0, 0.35355344414711, -0.3261861801147461, -0.2578643560409546, 1.0]
    # params['projMatPanda'] = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    # params['cameraUp'] = [0.0, 0.0, 1.0]
    # params['camForward'] = [0.3535534143447876, -0.35355353355407715, -0.8660253882408142]
    # params['horizon'] = [-18856.18359375, -18856.177734375, 0.0]
    # params['vertical'] = [12247.4462890625, -12247.4501953125, 10000.001953125]
    # params['dist'] = 0.4000000059604645
    # params['camTarget'] = [0.5, 0.0, 0.03999999910593033]

    # p.resetDebugVisualizerCamera(0.40, 225, -45, [0.50, 0.0, 0.02])  # 40cm away
    params['viewMatPanda'] = [-0.7071067690849304, 0.5, -0.4999999701976776, 0.0, -0.7071067690849304, -0.5, 0.5, 0.0, 0.0, 0.7071067094802856, 0.7071068286895752, 0.0, 0.3535534143447876, -0.26414212584495544, -0.1641421914100647, 1.0]
    params['projMatPanda'] = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    params['cameraUp'] = [0.0, 0.0, 1.0]
    params['camForward'] = [0.4999999701976776, -0.5, -0.7071068286895752]
    params['horizon'] = [-18856.181640625, -18856.181640625, 0.0]
    params['vertical'] = [10000.0, -10000.0009765625, 14142.134765625]
    params['dist'] = 0.4000000059604645
    params['camTarget'] = [0.5, 0.0, 0.019999999552965164]

    # width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
    # print(width)
    # print(height)
    # print(viewMat)
    # print(projMat)
    # print(cameraUp)
    # print(camForward)
    # print(horizon)
    # print(vertical)
    # print(dist)
    # print(camTarget)

    # params['viewMatPanda'] = viewMat
    # params['projMatPanda'] = projMat
    # params['cameraUp'] = cameraUp
    # params['camForward'] = camForward
    # params['horizon'] = horizon
    # params['vertical'] = vertical
    # params['dist'] = dist
    # params['camTarget'] = camTarget

    ###########################################################################

    # calculations of near and far based on projection matrix
    # https://answers.unity.com/questions/1359718/what-do-the-values-in-the-matrix4x4-for-cameraproj.html
    # https://forums.structure.io/t/near-far-value-from-projection-matrix/3757
    m22 = params['projMatPanda'][10]
    m32 = params['projMatPanda'][14]  # THe projection matrix (array[15]) returned by PyBullet orders using column first
    params['near'] = 2*m32/(2*m22-2)
    params['far'] = ((m22-1.0)*params['near'])/(m22+1.0)

    print('Near', params['near'])
    print('Far: ', params['far'])

    return params

# Camera pixel location to 3D world location (point cloud)
def pixelToWorld(depthBuffer, jitterDepth=True):
    point_cloud = np.zeros((0,3))
    params = getParameters()
    far = params['far']  # 998.6
    near = params['near']  # 0.01
    # near = 0.01
    # far = 1000

    stepX = 1
    stepY = 1
    for w in range(168, 168+128, stepX):
        for h in range(176, 176+128, stepY):
            rayFrom, rayTo, alpha = getRayFromTo(w, h)
            rf = np.array(rayFrom)
            rt = np.array(rayTo)
            vec = rt - rf
            l = np.sqrt(np.dot(vec, vec))
            depthImg = float(depthBuffer[h, w])
            depth = far * near / (far - (far - near) * depthImg)

            if jitterDepth:
                depth += np.random.normal(0, 0.0005, 1)[0]

            depth /= math.cos(alpha)
            newTo = (depth / l) * vec + rf

            if newTo[2] > 0.003:
                point_cloud = np.concatenate((point_cloud, newTo.reshape(1,3)), axis=0)

    # Keep only 1000 table points
    # table_chosen = list(np.random.choice(table_ind, 1000, replace=False))
    # table_remove_ind = list(set(table_ind) - set(table_chosen))
    # point_cloud = np.delete(point_cloud, table_remove_ind, axis=0)
    return point_cloud


def getRayFromTo(mouseX, mouseY):
    params = getParameters()

    width = params['imgW']
    height = params['imgH']
    # cameraUp = params['cameraUp']
    camForward = params['camForward']
    horizon = params['horizon']
    vertical = params['vertical']
    dist = params['dist']
    camTarget = params['camTarget']

    camPos = [
        camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
        camTarget[2] - dist * camForward[2]
    ]
    farPlane = 10000
    rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
    lenFwd = math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] +
                        rayForward[2] * rayForward[2])
    invLen = farPlane * 1. / lenFwd
    rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
    rayFrom = camPos
    oneOverWidth = float(1) / float(width)
    oneOverHeight = float(1) / float(height)

    dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
    dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
    # rayToCenter = [
        # rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
    # ]
    ortho = [
        -0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] - float(mouseY) * dVer[0],
        -0.5 * horizon[1] + 0.5 * vertical[1] + float(mouseX) * dHor[1] - float(mouseY) * dVer[1],
        -0.5 * horizon[2] + 0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
    ]

    rayTo = [
        rayFrom[0] + rayForward[0] + ortho[0], rayFrom[1] + rayForward[1] + ortho[1],
        rayFrom[2] + rayForward[2] + ortho[2]
    ]
    lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2])
    alpha = math.atan(lenOrtho / farPlane)
    return rayFrom, rayTo, alpha

# def estimateCOM(img, depth):

#     imgx = np.tile(range(0, img.shape[1]), (img.shape[0],1))
#     imgy = np.tile(np.reshape(range(0, img.shape[0]), (img.shape[0], 1)), (1, img.shape[1]))

#     xcom_img = np.sum(np.multiply(imgx,img))/np.sum(img)
#     ycom_img = np.sum(np.multiply(imgy,img))/np.sum(img)
#     # print(xcom_img)
#     # print(ycom_img)
#     xcom, ycom = pixelToWorld(xcom_img, ycom_img, depth)

#     return xcom, ycom


    # p.resetDebugVisualizerCamera(0.40, 90, -60, [0.46, 0.0, 0.04])  # 40cm away
    # params['viewMatPanda'] = [0.0, -0.8660253286361694, 0.49999988079071045, 0.0, 0.9999999403953552, 0.0, -0.0, 0.0, -0.0, 0.49999985098838806, 0.8660253882408142, 0.0, -0.0, 0.3783717155456543, -0.6646409034729004, 1.0]
    # params['projMatPanda'] = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    # params['cameraUp'] = [0.0, 0.0, 1.0]
    # params['camForward'] = [-0.49999988079071045, 0.0, -0.8660253882408142]
    # params['horizon'] = [0.0, 26666.66796875, -0.0]
    # params['vertical'] = [-17320.509765625, 0.0, 9999.9990234375]
    # params['dist'] = 0.4000000059604645
    # params['camTarget'] = [0.46000000834465027, 0.0, 0.03999999910593033]

    # p.resetDebugVisualizerCamera(0.40, 225, -60, [0.46, 0.0, 0.04])  # 40cm away
    # params['viewMatPanda'] = [-0.7071070075035095, 0.6123722791671753, -0.35355332493782043, 0.0, -0.7071065902709961, -0.612372636795044, 0.35355353355407715, 0.0, 0.0, 0.5000000596046448, 0.8660253882408142, 0.0, 0.3252692222595215, -0.3016912639141083, -0.2720065116882324, 1.0]
    # params['projMatPanda'] = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    # params['cameraUp'] = [0.0, 0.0, 1.0]
    # params['camForward'] = [0.35355332493782043, -0.35355353355407715, -0.8660253882408142]
    # params['horizon'] = [-18856.185546875, -18856.17578125, 0.0]
    # params['vertical'] = [12247.4453125, -12247.451171875, 10000.0009765625]
    # params['dist'] = 0.4000000059604645
    # params['camTarget'] = [0.46000000834465027, 0.0, 0.03999999910593033]
