import numpy as np

# Number of keyframes in the cycle
K = 6

# Starter keyframes from kf-halfcheetah.py (used as one individual in the initial population)
# Each keyframe: [torque1, torque2, torque3, torque4, torque5, torque6, duration]
starter_keyframes = [
    np.array([ 0.9, -0.7,  0.6, -0.6,  0.5, -0.3,  10]),
    np.array([ 0.5, -0.9,  0.4,  0.5, -0.6,  0.7,  10]),
    np.array([ 0.3, -0.5,  0.2,  0.8, -0.8,  0.5,   8]),
    np.array([-0.6,  0.5, -0.3,  0.9, -0.7,  0.6,  10]),
    np.array([ 0.5, -0.6,  0.7,  0.5, -0.9,  0.4,  10]),
    np.array([ 0.8, -0.8,  0.5,  0.3, -0.5,  0.2,   8]),
]

# EA hyperparameters
POPULATION_SIZE = 20    # more diversity
NUM_OFFSPRING   = 40    # more children per generation
NUM_GENERATIONS = 5   # more time to improve
