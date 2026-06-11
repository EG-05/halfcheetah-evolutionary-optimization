import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy

from config  import K, starter_keyframes, POPULATION_SIZE, NUM_OFFSPRING, NUM_GENERATIONS
from fitness import evaluate
from ea      import (run_ea, initialise_population,
                     tournament_selection, roulette_selection, rank_selection,
                     truncation_survival, random_survival, fitness_proportionate_survival)

# Six combinations of parent selection + survival selection to compare
EXPERIMENTS = [
    ("Tournament + Truncation", tournament_selection, truncation_survival),
    ("Roulette + Truncation", roulette_selection, truncation_survival),
    ("Rank + Truncation", rank_selection, truncation_survival),
    ("Tournament + Random", tournament_selection, random_survival),
    ("Roulette + Fitness-proportionate",roulette_selection, fitness_proportionate_survival),
    ("Rank + Fitness-proportionate",rank_selection, fitness_proportionate_survival),
]


def main():
    env = gym.make("HalfCheetah-v5")  

    all_histories       = []
    all_labels          = []
    overall_best_chr    = None
    overall_best_reward = -np.inf

    for label, parent_sel, survival_sel in EXPERIMENTS:
        print(f"\n── {label} ──")

        population = initialise_population(POPULATION_SIZE, K, starter_keyframes)

        best_chr, history = run_ea(
            initial_population = population,
            env                = env,
            num_generations    = NUM_GENERATIONS,
            num_offspring      = NUM_OFFSPRING,
            parent_selection   = parent_sel,
            survival_selection = survival_sel,
        )

        all_histories.append(history)
        all_labels.append(label)
        print(f"  Final best reward: {history[-1]:.2f}")

        if history[-1] > overall_best_reward:
            overall_best_reward = history[-1]
            overall_best_chr    = copy.deepcopy(best_chr)

    env.close()

    # Save best keyframes 
    with open("best_keyframes.txt", "w") as f:
        f.write(f"{len(overall_best_chr)}\n")
        for kf in overall_best_chr:
            values   = ", ".join(f"{v:.2f}" for v in kf[:-1])
            duration = int(round(kf[-1]))
            f.write(f"{values}, {duration}\n")
    print("\n✓ Saved best_keyframes.txt")

    # Convergence plot 
    plt.figure(figsize=(10, 6))
    for history, label in zip(all_histories, all_labels):
        plt.plot(history, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Best Episode Reward")
    plt.title("Convergence Plot — EA Keyframe Optimisation (HalfCheetah-v5)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150)
    plt.close()
    print("Saved convergence_plot.png")

    # Print final rewards 
    print("\n── Final Rewards ──")
    for label, history in zip(all_labels, all_histories):
        print(f" {label:<45} {history[-1]:.2f}")
    print(f"\n Overall best reward: {overall_best_reward:.2f}")

    print("\nVisualising best keyframes …")
    vis_env = gym.make("HalfCheetah-v5", render_mode="human", width=1280, height=720)
    observation, info   = vis_env.reset(seed=42)
    keyframe_index      = 0
    remaining_steps     = int(overall_best_chr[keyframe_index][-1])
    episode_reward      = 0

    while True:
        action = overall_best_chr[keyframe_index][:-1]
        observation, reward, terminated, truncated, info = vis_env.step(action)
        episode_reward  += reward
        remaining_steps -= 1
        if remaining_steps == 0:
            keyframe_index  = (keyframe_index + 1) % len(overall_best_chr)
            remaining_steps = int(overall_best_chr[keyframe_index][-1])
        if terminated or truncated:
            print(f"Episode ended. Total reward: {episode_reward:.2f}")
            # Reset and keep going so the window stays open
            observation, info = vis_env.reset()
            keyframe_index    = 0
            remaining_steps   = int(overall_best_chr[keyframe_index][-1])
            episode_reward    = 0


if __name__ == "__main__":
    main()
