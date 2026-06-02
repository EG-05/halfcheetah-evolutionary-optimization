import numpy as np

# Number of keyframes in the cycle
K = 5

# Starter keyframes from kf-halfcheetah.py (used as one individual in the initial population)
# Each keyframe: [torque1, torque2, torque3, torque4, torque5, torque6, duration]
starter_keyframes = [
    np.array([-0.10,  0.05, -0.08,  0.12, -0.06,  0.03, 25]),
    np.array([ 0.20, -0.15,  0.30, -0.10,  0.25, -0.05, 40]),
    np.array([ 0.40, -0.40,  0.50, -0.30,  0.50, -0.20, 15]),
    np.array([ 0.50, -0.20,  0.40, -0.15,  0.30, -0.10, 20]),
    np.array([ 0.10,  0.00, -0.10,  0.05, -0.05,  0.02, 30]),
]

# EA hyperparameters
POPULATION_SIZE = 20    # more diversity
NUM_OFFSPRING   = 40    # more children per generation
NUM_GENERATIONS = 5   # more time to improve
