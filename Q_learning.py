# Q_learning.py
# Q-learning implementation for the museum-heist task.
# Includes training, experiments, and visualization helpers.

"""
Q-learning Agent and Experiments (Q_learning.py)
==================================================
Contains the Q-learning training loop, both required experiments,
and all result visualizations for the Museum Thief project.

KEY FUNCTIONS:
    train_q_learning(env, no_episodes, ...)  — trains the agent
    visualize_q_table(q_values_path)         — draws policy maps
    experiment_exploration(env_factory, ...) — Exp1: epsilon strategies
    experiment_learning_rates(env_factory, ...)— Exp2: alpha comparison
    plot_coverage(visit_counts)              — exploration heatmap

All outputs are saved to the results/ folder.
"""


# Imports
import os
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import grid constants used for visualization.
from padm_env import GRID, GRID_SIZE, REWARDS, CELL_NAME

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Map action indices to arrows for policy plots.
ACTION_ARROWS = {0: '↑', 1: '↓', 2: '→', 3: '←'}
ACTION_NAMES  = ['Up', 'Down', 'Right', 'Left']

# Colors for the museum map cells.
CELL_COLORS = {
    'S': '#9ecae1', 'D': '#ffd700', 'E': '#74c476',
    'G': '#fdae6b', 'T': '#bcbddc', 'C': '#fdd0a2',
    'R': '#252525', 'A': '#cb181d', '.': '#f7f7f7',
}

# Train the Q-learning agent.
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     render,
                     fps=10,
                     q_table_save_path=None,
                     return_history=False,
                     snapshot_checkpoints=None,
                     snapshot_store=None):
    """
    Train the museum thief agent using Q-learning.
    Returns q_table (and optionally history dict for experiments).
    """
    if q_table_save_path is None:
        q_table_save_path = os.path.join(RESULTS_DIR, 'q_table.npy')

    # Q-table has shape (11,11,2,4) to include the carrying flag.
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, 2, env.action_space.n))

    snapshot_checkpoints = set(snapshot_checkpoints or [])
    if snapshot_store is not None and 0 in snapshot_checkpoints:
        snapshot_store.append(q_table.copy())

    # Track episode rewards, epsilon values, wins, and visit counts.
    ep_rewards  = []
    ep_epsilons = []
    ep_wins     = []
    visit_counts = np.zeros((GRID_SIZE, GRID_SIZE, 2))  # for coverage map

    # Step 1: Run for fixed number of episodes (same as template)
    for episode in range(no_episodes):
        obs, _ = env.reset()
        # Unpack the observation as [row, col, carrying].
        r, c, carrying = int(obs[0]), int(obs[1]), int(obs[2])
        total_reward = 0
        won = 0

        # Step 2: Take actions until done (same structure as template)
        while True:
            visit_counts[r, c, carrying] += 1

            # Explore or exploit based on epsilon.
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                # Choose the greedy action from the Q-table.
                action = np.argmax(q_table[r, c, carrying])

            obs_next, reward, done, info = env.step(action)

            if render:
                extra = (f"Ep:{episode+1}/{no_episodes}  "
                         f"ε:{epsilon:.3f}  R:{total_reward:.0f}")
                env.render(fps=fps, extra_info=extra)

            # Unpack the next observation.
            nr, nc, n_carrying = int(obs_next[0]), int(obs_next[1]), int(obs_next[2])
            total_reward += reward

            # Update the Q-value using the Bellman equation.
            best_next = np.max(q_table[nr, nc, n_carrying])
            q_table[r, c, carrying, action] = (
                q_table[r, c, carrying, action] +
                alpha * (reward + gamma * best_next
                         - q_table[r, c, carrying, action])
            )

            r, c, carrying = nr, nc, n_carrying

            # Step 5: Stop if done (same as template)
            if done:
                if info.get('event') == 'escaped_win':
                    won = 1
                break

        ep_rewards.append(total_reward)
        ep_epsilons.append(epsilon)
        ep_wins.append(won)

        if snapshot_store is not None and (episode + 1) in snapshot_checkpoints:
            snapshot_store.append(q_table.copy())

        # Step 6: Epsilon decay (same as template)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0 or episode == 0:
            win_rate = np.mean(ep_wins[-200:]) * 100
            print(f"  Ep {episode+1:5d}/{no_episodes} | "
                  f"eps={epsilon:.4f} | "
                  f"avg_R={np.mean(ep_rewards[-200:]):6.1f} | "
                  f"win%={win_rate:.1f}")

    # Step 7: Close env (same as template — caller closes if render=True)
    if not render:
        env.close()

    # Step 8: Save Q-table (same as template)
    np.save(q_table_save_path, q_table)
    print(f"\n  Q-table saved -> {q_table_save_path}")

    history = {
        'rewards':      np.array(ep_rewards),
        'epsilons':     np.array(ep_epsilons),
        'wins':         np.array(ep_wins),
        'visit_counts': visit_counts,
    }

    if return_history:
        return q_table, history
    return q_table


# Visualize the learned Q-table for both phases.
def visualize_q_table(q_values_path=None):
    if q_values_path is None:
        q_values_path = os.path.join(RESULTS_DIR, 'q_table.npy')

    try:
        q_table = np.load(q_values_path)
    except FileNotFoundError:
        print("No Q-table found. Train first.")
        return

    # Hide terminal states in the heatmaps.
    terminal = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for rr in range(GRID_SIZE):
        for cc in range(GRID_SIZE):
            if GRID[rr][cc] in ('R', 'A', 'E', 'D'):
                terminal[rr, cc] = True

    phase_labels = ['Phase 0 — not carrying (goal: Diamond)',
                    'Phase 1 — carrying diamond (goal: Exit)']

    for phase in range(2):
        # ---- per-action heatmaps (same as template visualize_q_table) ----
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        for i, action in enumerate(ACTION_NAMES):
            ax = axes[i]
            # === MUSEUM CHANGE: slice Q-table for this phase ===
            heatmap_data = q_table[:, :, phase, i].copy()
            mask = terminal.copy()
            sns.heatmap(heatmap_data, annot=True, fmt='.1f',
                        cmap='RdYlGn', ax=ax, cbar=(i == 3),
                        mask=mask, annot_kws={'size': 7},
                        linewidths=0.4, linecolor='white',
                        vmin=-25, vmax=100)
            # Mark special cells (same approach as template)
            for rr in range(GRID_SIZE):
                for cc in range(GRID_SIZE):
                    sym = GRID[rr][cc]
                    if sym not in ('.', 'S'):
                        ax.text(cc+0.5, rr+0.5, sym,
                                color='white' if sym in ('R','A') else 'black',
                                ha='center', va='center',
                                fontsize=8, fontweight='bold')
            ax.set_title(f'Action: {action}', fontsize=11)
            ax.set_xlabel('Column'); ax.set_ylabel('Row')

        fig.suptitle(f'Q-values per action — {phase_labels[phase]}', fontsize=13)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f'qtable_actions_phase{phase}.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"  -> saved {fname}")

        # ---- best-action (policy) heatmap (same as template) ----
        best_actions  = np.argmax(q_table[:, :, phase, :], axis=2)
        best_q_values = np.max(q_table[:, :, phase, :], axis=2)

        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(best_q_values, annot=False, cmap='RdYlGn',
                    ax=ax, cbar=True, mask=terminal,
                    linewidths=0.4, linecolor='white',
                    vmin=-25, vmax=100)
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                sym = GRID[rr][cc]
                if terminal[rr, cc]:
                    ax.text(cc+0.5, rr+0.5, sym,
                            color='white' if sym in ('R','A') else 'black',
                            ha='center', va='center',
                            fontsize=10, fontweight='bold')
                else:
                    arrow = ACTION_ARROWS[best_actions[rr, cc]]
                    qval  = best_q_values[rr, cc]
                    ax.text(cc+0.5, rr+0.3, arrow,
                            color='white', ha='center', va='center',
                            fontsize=14, fontweight='bold')
                    ax.text(cc+0.5, rr+0.75, f'{qval:.0f}',
                            color='white', ha='center', va='center',
                            fontsize=7)
        ax.set_title(f'Optimal policy + max Q-value\n{phase_labels[phase]}',
                     fontsize=12)
        ax.set_xlabel('Column'); ax.set_ylabel('Row')
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f'policy_phase{phase}.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        # === MUSEUM CHANGE: show chart on screen after saving ===
        plt.show()
        plt.close(fig)
        print(f"  -> saved {fname}")


# ===========================================================================
# Function 3: Plot training curves
# === MUSEUM CHANGE: New function — saves reward, win-rate, epsilon curves ===
# ===========================================================================
def plot_training_curves(history, label='default', color='steelblue',
                         axes=None, show_legend=True):
    """Plot reward, win-rate, and epsilon curves from a training history dict."""
    rewards  = history['rewards']
    epsilons = history['epsilons']
    wins     = history['wins']
    episodes = np.arange(1, len(rewards) + 1)
    window   = min(200, len(rewards) // 10)

    def smooth(x):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        standalone = True
    else:
        standalone = False

    axes[0].plot(episodes[window-1:], smooth(rewards),
                 label=label, color=color, lw=2)
    axes[0].set_title('Smoothed Reward per Episode')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes[window-1:], smooth(wins) * 100,
                 label=label, color=color, lw=2)
    axes[1].set_title('Win Rate % (smoothed)')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Win %')
    axes[1].set_ylim(0, 105); axes[1].grid(alpha=0.3)

    axes[2].plot(episodes, epsilons, label=label, color=color, lw=2)
    axes[2].set_title('Epsilon Decay Curve')
    axes[2].set_xlabel('Episode'); axes[2].set_ylabel('ε')
    axes[2].set_ylim(0, 1.05); axes[2].grid(alpha=0.3)

    if show_legend:
        for ax in axes:
            ax.legend(fontsize=9)

    if standalone:
        fig.suptitle('Training Curves', fontsize=14)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, 'training_curves.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        # === MUSEUM CHANGE: show chart on screen after saving ===
        plt.show()
        plt.close(fig)
        print(f"  -> saved {fname}")
        return None
    return axes


# ===========================================================================
# Function 4: Experiment 1 — compare exploration strategies (ε-decay)
# === MUSEUM CHANGE: Required deliverable — at least 2 strategies ===
# ===========================================================================
def experiment_exploration(env_factory, no_episodes=5000,
                           alpha=0.1, gamma=0.99):
    """
    Compare TWO epsilon-decay strategies (deliverable requirement):
      - Exponential decay: eps × decay_rate each episode
      - Linear decay:      eps decreases by fixed step each episode
    Both use same alpha and gamma.
    """
    print("\n=== Experiment 1: Exploration Strategies (epsilon decay) ===")
    print("Fixed: alpha=0.1, gamma=0.99. Comparing exponential vs linear decay.")

    strategies = {
        'Exponential (eps x 0.995)': {
            'epsilon': 1.0, 'epsilon_min': 0.05,
            'epsilon_decay': 0.995, 'mode': 'exp'},
        'Linear (eps - 1/N)': {
            'epsilon': 1.0, 'epsilon_min': 0.05,
            'epsilon_decay': None, 'mode': 'lin'},
    }

    colors = ['#2171b5', '#e6550d']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Experiment 1 — Exploration Strategy Comparison\n'
                 'Fixed: alpha=0.1, gamma=0.99', fontsize=13)

    for (label, cfg), color in zip(strategies.items(), colors):
        print(f"\n  Training: {label}")
        env = env_factory(render_mode=False)

        # === MUSEUM CHANGE: Linear decay implemented manually ===
        if cfg['mode'] == 'lin':
            # Compute per-episode linear step
            eps_range  = cfg['epsilon'] - cfg['epsilon_min']
            decay_step = eps_range / no_episodes

            q_table = np.zeros((GRID_SIZE, GRID_SIZE, 2, env.action_space.n))
            ep_rewards, ep_eps, ep_wins = [], [], []
            eps = cfg['epsilon']

            for episode in range(no_episodes):
                obs, _ = env.reset()
                r, c, carrying = int(obs[0]), int(obs[1]), int(obs[2])
                total_reward = 0; won = 0
                while True:
                    if np.random.rand() < eps:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(q_table[r, c, carrying])
                    obs_n, reward, done, info = env.step(action)
                    nr, nc, nc2 = int(obs_n[0]), int(obs_n[1]), int(obs_n[2])
                    total_reward += reward
                    best_next = np.max(q_table[nr, nc, nc2])
                    q_table[r,c,carrying,action] += alpha*(
                        reward + gamma*best_next - q_table[r,c,carrying,action])
                    r, c, carrying = nr, nc, nc2
                    if done:
                        if info.get('event') == 'escaped_win': won = 1
                        break
                ep_rewards.append(total_reward)
                ep_eps.append(eps)
                ep_wins.append(won)
                eps = max(cfg['epsilon_min'], eps - decay_step)
                if (episode+1) % 1000 == 0:
                    print(f"    ep {episode+1} | eps={eps:.4f} | "
                          f"win%={np.mean(ep_wins[-200:])*100:.1f}")
            env.close()
            history = {'rewards': np.array(ep_rewards),
                       'epsilons': np.array(ep_eps),
                       'wins': np.array(ep_wins)}
        else:
            # Exponential — use main train function
            q_table, history = train_q_learning(
                env=env, no_episodes=no_episodes,
                epsilon=cfg['epsilon'], epsilon_min=cfg['epsilon_min'],
                epsilon_decay=cfg['epsilon_decay'],
                alpha=alpha, gamma=gamma,
                render=False, return_history=True)

        axes = plot_training_curves(history, label=label,
                                    color=color, axes=axes,
                                    show_legend=False)

        final_wr = np.mean(history['wins'][-200:]) * 100
        print(f"  {label}: final win rate = {final_wr:.1f}%")

    for ax in axes:
        ax.legend(fontsize=9)

    fig.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'exp1_exploration_strategies.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    # === MUSEUM CHANGE: show chart on screen after saving ===
    plt.show()
    plt.close(fig)
    print(f"\n  -> saved {fname}")

    # Explanation saved alongside
    with open(os.path.join(RESULTS_DIR, 'exp1_explanation.txt'), 'w', encoding="utf-8") as f:
        f.write(
            "EXPERIMENT 1 — EXPLORATION STRATEGIES\n"
            "======================================\n"
            "We compare two epsilon-decay schedules (deliverable: >=2):\n\n"
            "1. Exponential decay: ε = ε × 0.995 each episode.\n"
            "   Fast early decay, long gentle tail.\n"
            "   Best for environments where good routes are discovered quickly.\n\n"
            "2. Linear decay: ε = ε - (1/N) each episode.\n"
            "   Steady, predictable reduction.\n"
            "   Gives equal exploration time to all episodes proportionally.\n\n"
            "Theory justification (from lecture):\n"
            "High epsilon early = exploration (agent knows nothing).\n"
            "Low epsilon later  = exploitation (agent uses what it learned).\n"
            "The optimal schedule decays fast enough to converge but slow\n"
            "enough that every reachable state gets visited early.\n"
        )
    print("  -> saved exp1_explanation.txt")


# ===========================================================================
# Function 5: Experiment 2 — compare learning rates (α)
# === MUSEUM CHANGE: Required deliverable — at least 3 learning rates ===
# ===========================================================================
def experiment_learning_rates(env_factory, no_episodes=5000,
                               epsilon_decay=0.995, gamma=0.99):
    """
    Compare THREE alpha values (deliverable requirement):
      α = 0.01, 0.1, 0.3
    Fixed: exponential ε-decay, γ=0.99.
    """
    print("\n=== Experiment 2: Learning Rate (alpha) Comparison ===")
    print("Fixed: exponential eps-decay=0.995, gamma=0.99. Comparing alpha values.")

    alphas = [0.01, 0.1, 0.3]
    colors = ['#2ca02c', '#1f77b4', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Experiment 2 — Learning Rate (α) Comparison\n'
                 'Fixed: exponential ε-decay=0.995, γ=0.99', fontsize=13)

    for alpha, color in zip(alphas, colors):
        print(f"\n  Training: alpha = {alpha}")
        env = env_factory(render_mode=False)
        _, history = train_q_learning(
            env=env, no_episodes=no_episodes,
            epsilon=1.0, epsilon_min=0.05,
            epsilon_decay=epsilon_decay,
            alpha=alpha, gamma=gamma,
            render=False, return_history=True)
        axes = plot_training_curves(history, label=f'α={alpha}',
                                    color=color, axes=axes,
                                    show_legend=False)
        final_wr = np.mean(history['wins'][-200:]) * 100
        print(f"  alpha={alpha}: final win rate = {final_wr:.1f}%")

    for ax in axes:
        ax.legend(fontsize=9)

    fig.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'exp2_learning_rates.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    # === MUSEUM CHANGE: show chart on screen after saving ===
    plt.show()
    plt.close(fig)
    print(f"\n  -> saved {fname}")

    with open(os.path.join(RESULTS_DIR, 'exp2_explanation.txt'), 'w', encoding="utf-8") as f:
        f.write(
            "EXPERIMENT 2 — LEARNING RATE (α) COMPARISON\n"
            "=============================================\n"
            "We compare three alpha values (deliverable: >=3):\n"
            "  α = 0.01 (small), 0.1 (medium), 0.3 (large)\n\n"
            "Alpha controls the step size of each Q-value update:\n"
            "  Q(s,a) ← Q(s,a) + α × [target - Q(s,a)]\n\n"
            "Small α (0.01): learns slowly but stably.\n"
            "  Each experience has little influence — needs many episodes.\n\n"
            "Medium α (0.1): balanced speed and stability.\n"
            "  Recommended default for deterministic environments.\n\n"
            "Large α (0.3): learns fast but can overshoot.\n"
            "  Recent experience dominates; may oscillate around optimum.\n\n"
            "Theory justification (from lecture):\n"
            "In a deterministic environment (our museum), each observation is\n"
            "reliable so larger alpha is safe. In stochastic environments a\n"
            "smaller alpha averages out noise — but here 0.1 is the sweet spot.\n"
        )
    print("  -> saved exp2_explanation.txt")


# ===========================================================================
# Function 6: Coverage / exploration heatmap
# === MUSEUM CHANGE: New — shows which cells were visited during training ===
# ===========================================================================
def plot_coverage(visit_counts):
    """Show how many times each cell was visited — confirms full exploration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    phase_labels = ['Phase 0 (not carrying)', 'Phase 1 (carrying)']

    for phase, ax in enumerate(axes):
        data = visit_counts[:, :, phase].copy()
        # Use sqrt scaling so rarely-visited cells still show color
        sns.heatmap(np.sqrt(data), annot=False, cmap='magma',
                    ax=ax, cbar=True, linewidths=0.3, linecolor='gray')
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                n = int(data[rr, cc])
                sym = GRID[rr][cc]
                ax.text(cc+0.5, rr+0.35, f'{n:,}',
                        color='white', ha='center', va='center',
                        fontsize=7, fontweight='bold')
                if sym not in ('.', 'S'):
                    ax.text(cc+0.5, rr+0.72, sym,
                            color='yellow', ha='center', va='center',
                            fontsize=8)
        ax.set_title(f'Visit counts — {phase_labels[phase]}')
        ax.set_xlabel('Column'); ax.set_ylabel('Row')

    reached = int(np.sum(visit_counts.sum(axis=2) > 0))
    fig.suptitle(f'Exploration coverage (cells reached: {reached}/{GRID_SIZE*GRID_SIZE})\n'
                 'bright = visited often, dark = rarely visited', fontsize=13)
    fig.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'coverage_heatmap.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    # === MUSEUM CHANGE: show chart on screen after saving ===
    plt.show()
    plt.close(fig)
    print(f"  -> saved {fname}")
    print(f"  Cells reached: {reached}/{GRID_SIZE*GRID_SIZE}")