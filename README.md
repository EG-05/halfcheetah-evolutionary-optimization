# HalfCheetah-v5 Optimization using Evolutionary Algorithms

The objective was to optimize the locomotion of the HalfCheetah-v5 agent in the MuJoCo simulation environment using Evolutionary Algorithms (EA).

Rather than learning a policy through reinforcement learning, the agent's movement is represented as a sequence of keyframes. Evolutionary operators are then used to evolve increasingly effective motion patterns that maximize cumulative episode reward.

## Environment

* Environment: HalfCheetah-v5
* Simulator: MuJoCo
* Interface: Gymnasium
* Language: Python

At each timestep, the environment accepts a 6-dimensional torque vector and returns a reward based on forward velocity with penalties for energy consumption.

## Keyframe Representation

Each individual in the population is represented as:

(K, 7)

Where:

* K = number of keyframes
* First 6 values = joint torque commands
* Final value = duration for which the keyframe is held

During evaluation, keyframes are executed cyclically until the episode terminates.

## Fitness Function

Fitness is defined as the cumulative reward obtained during a complete episode:

* Higher forward velocity increases fitness
* Excessive energy usage reduces fitness

A fixed random seed is used during evaluation to ensure fair comparisons between candidate solutions.

## Evolutionary Operators

### Crossover

#### One-Point Crossover
A random split point is selected and keyframes are exchanged between parents.

#### Uniform Crossover
Each keyframe is independently inherited from either parent with equal probability.

#### Arithmetic Crossover
Children are generated through weighted averaging of parent genes, making it particularly suitable for continuous control parameters.

### Mutation

#### Gaussian Mutation
Adds normally distributed noise to selected genes.

#### Uniform Reset Mutation
Randomly replaces selected genes with newly generated values from the search space.

## Selection Methods

### Parent Selection

* Tournament Selection
* Roulette-Wheel Selection
* Rank-Based Selection

### Survival Selection

* Truncation (μ + λ)
* Random Survival (μ, λ)
* Fitness-Proportionate Survival

## Experimental Evaluation

Six combinations of parent-selection and survival-selection strategies were tested to compare:

* Convergence speed
* Population diversity
* Exploration vs exploitation
* Final fitness achieved

Examples include:

* Tournament + Truncation
* Roulette + Truncation
* Rank + Truncation
* Tournament + Random
* Roulette + Fitness-Proportionate
* Rank + Fitness-Proportionate

## Results

The project generated:

### Best Keyframe Solution
`best_keyframes.txt`
Stores the highest-performing evolved motion sequence.

### Convergence Analysis
`convergence_plot.png`

Displays best-fitness and average-fitness trends across generations for all experimental configurations.

## Key Takeaways

* Strong elitist strategies often converged faster but reduced population diversity.
* More exploratory approaches maintained diversity and were less susceptible to premature convergence.
* Arithmetic crossover performed particularly well for continuous-valued control parameters.
* Evolutionary Algorithms can successfully discover effective locomotion behaviours without explicitly defining how the agent should move.

## Technologies Used

* Python
* NumPy
* Gymnasium
* MuJoCo
* Evolutionary Algorithms

## Learning Outcomes

This project demonstrates how evolutionary computation can be applied to continuous control problems and highlights the impact of selection pressure, diversity preservation, crossover design, and mutation strategies on optimization performance.
