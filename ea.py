import numpy as np
import copy
from fitness import evaluate


# A chromosome is a list of K keyframes.
# Each keyframe is a numpy array [torque1, torque2, torque3, torque4, torque5, torque6, duration]

def random_chromosome(k):

    chromosome = []

    for _ in range(k):
        torques  = np.random.uniform(-0.35, 0.35, 6)
        duration = float(np.random.randint(5, 51))
        chromosome.append(np.append(torques, duration))
    return chromosome


def initialise_population(size, k, starter_chromosome):
    
    #Build the starting population, slot 0 = starter chromosome 
    
    population = [copy.deepcopy(starter_chromosome)]
    for _ in range(size - 1):
        population.append(random_chromosome(k))
    return population


# Parent selection (Pre-selection) 

def tournament_selection(population, fitnesses, k=3):

    #Pick k random individuals and return the best one
    contestants = np.random.choice(len(population), k, replace=False)
    winner      = max(contestants, key=lambda i: fitnesses[i])
    return population[winner]


def roulette_selection(population, fitnesses):

    #Pick one individual with probability proportional to its fitness
    scores = np.array(fitnesses) - min(fitnesses)
    total  = scores.sum()

    if total == 0:
        return population[np.random.randint(len(population))]
    
    probs = scores / total
    return population[np.random.choice(len(population), p=probs)]


def rank_selection(population, fitnesses):

    #Pick one individual with probability proportional to its rank
    order = np.argsort(fitnesses)
    ranks = np.empty(len(population))

    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1

    probs = ranks / ranks.sum()
    return population[np.random.choice(len(population), p=probs)]



def crossover(parent1, parent2):

    #One-point crossover -> swap keyframes after a random split point
    point  = np.random.randint(1, len(parent1))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome):

    # Adjust one gene in one randomly chosen keyframe
    mutated = copy.deepcopy(chromosome)
    i = np.random.randint(len(mutated))         
    j = np.random.randint(len(mutated[i]))      

    if j == len(mutated[i]) - 1:
        # duration gene -> change by small integer
        mutated[i][j] += np.random.randint(-5, 6)
        mutated[i][j]  = np.clip(mutated[i][j], 5, 50)
    else:
        # torque gene —> change by small amount
        mutated[i][j] += np.random.uniform(-0.2, 0.2)
        mutated[i][j]  = np.clip(mutated[i][j], -1.0, 1.0)

    return mutated


# Survivor Selection (Post-selection) 
# Sort by i as a tiebreaker to avoid comparing numpy arrays directly

def truncation_survival(population, fitnesses, mu):

    ranked        = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    survivors     = [population[i] for i in ranked[:mu]]
    survivor_fits = [fitnesses[i]  for i in ranked[:mu]]

    return survivors, survivor_fits


def random_survival(population, fitnesses, mu):

    indices       = np.random.choice(len(population), mu, replace=False)
    survivors     = [population[i] for i in indices]
    survivor_fits = [fitnesses[i]  for i in indices]

    return survivors, survivor_fits


def fitness_proportionate_survival(population, fitnesses, mu):
    #Keep mu individuals chosen with probability proportional to fitness
    scores = np.array(fitnesses) - min(fitnesses)
    total  = scores.sum()
    if total == 0:

        return random_survival(population, fitnesses, mu)
    
    probs         = scores / total
    indices       = np.random.choice(len(population), mu, replace=False, p=probs)
    survivors     = [population[i] for i in indices]
    survivor_fits = [fitnesses[i]  for i in indices]

    return survivors, survivor_fits




def run_ea(initial_population, env, num_generations, num_offspring,
           parent_selection, survival_selection):
    
    #Initialisation -> Fitness -> Parent Selection -> Crossover -> Mutation -> Survivor Selection
    #Elitism - best individual is carried into the next generation
    
    population = initial_population
    fitnesses  = [evaluate(ind, env) for ind in population]

    # Tracking the best
    best_idx        = int(np.argmax(fitnesses))
    best_chromosome = copy.deepcopy(population[best_idx])
    best_reward     = fitnesses[best_idx]
    history         = [best_reward]

    for generation in range(num_generations):
        offspring = []

        # Parent Selection -> Crossover -> Mutation
        while len(offspring) < num_offspring:
            parent1 = parent_selection(population, fitnesses)
            parent2 = parent_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))


        offspring_fits = [evaluate(ind, env) for ind in offspring]

        combined      = population + offspring
        combined_fits = fitnesses  + offspring_fits
        population, fitnesses = survival_selection(combined, combined_fits, len(population))

        # If the best was lost we put it back in place of the worst
        if max(fitnesses) < best_reward:
            worst_idx             = int(np.argmin(fitnesses))
            population[worst_idx] = copy.deepcopy(best_chromosome)
            fitnesses[worst_idx]  = best_reward
        else:
            best_idx        = int(np.argmax(fitnesses))
            best_chromosome = copy.deepcopy(population[best_idx])
            best_reward     = fitnesses[best_idx]

        history.append(best_reward)
        print(f"  Gen {generation + 1:2d} | best reward: {best_reward:.2f}")

    return best_chromosome, history