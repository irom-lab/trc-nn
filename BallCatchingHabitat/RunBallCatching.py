import os
import numpy as np
import sys
import glob
import random
import BallSimulator

if __name__ == '__main__':
    
    numSimulations = int(sys.argv[1])
    print(numSimulations)
    
    simulateEnvironments(numSimulations)

## numSim is the number of different environments in which you want to run the simulation.
## Coords is a 2D array, of tuples, of dimension numSim x TimeHorizon.
## The simulation may not last for the entire TimeHorizon duration since the ball may collide
## before that happens. Could potentially add physics in the future to add support for bouncing
## of walls etc.
## Each tuple represents <x,y> coordinates. x and y are displacements relative to the 
## point at which the ball is launched
## The first tuple in each row would be <0, 0>
## Each row of the Coords matrix is the precomputed trajectory of an instance of a ball.
def simulateEnvironments(numSim, Coords):
    
    envGlbs = sorted(glob.glob("data/scene_datasets/allData/*.glb")) 
    np.random.seed(seed)
    np.random.shuffle(envGlbs)
    
    for envNum in range(numSim): 
        
        ## Parameters 
        record = True
        seed = 11
        T_Horizon = 50

        randEnv = np.random.randint(0, len(envGlbs))
        glb = envGlbs[randEnv]
        print("Environment Number : ", (envNum + 1))
        
        BallSimulator.RunSimulation(envNum, record, glb, Coords[envNum], seed)
    
