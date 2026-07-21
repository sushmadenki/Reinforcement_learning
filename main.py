# main.py
# Museum Thief RL pipeline for the PADM assignment.
# Runs training, experiments, visualizations, live gameplay, and GIF export.

"""
HOW TO RUN THIS PROJECT
=======================

Prerequisites:
    pip install gymnasium pygame matplotlib seaborn numpy pillow

Project structure required:
    chid/
    ├── main.py          <- run this file
    ├── padm_env.py      <- museum environment
    ├── Q_learning.py    <- Q-learning training and experiments
    ├── custom_env.py    <- original course template
    ├── uploads/         <- all GIF and image assets (guard.gif, etc.)
    └── results/         <- output folder (auto-created on first run)

To run the existing trained model (live game only — no retraining):
    python main.py

To run full pipeline (train + experiments + analysis + live game + GIF):
    python main.py --train

Live game controls (pygame window):
    Q     quit
    Close window to continue to the next step

OUTPUT:
    All results are saved to the results/ folder automatically.
"""


# Imports
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import the museum environment and training helpers.
from padm_env import create_env, GRID, GRID_SIZE, CELL_SIZE, save_environment_preview
from Q_learning import (train_q_learning, visualize_q_table,
                        plot_training_curves, plot_coverage,
                        experiment_exploration, experiment_learning_rates,
                        RESULTS_DIR)


# Configuration

# What to run
_TRAIN          = '--train' in sys.argv
RUN_TRAINING    = _TRAIN   # only retrain if --train flag is passed
RUN_EXPERIMENTS = _TRAIN   # only run experiments if --train flag is passed
RUN_ANALYSIS    = _TRAIN   # only generate charts if --train flag is passed
RUN_LIVE_GAME   = True     # always show live game
SAVE_GIF        = _TRAIN   # only save GIF if --train flag is passed

# Training hyperparameters for the museum task
no_episodes   = 5000
alpha         = 0.1      # learning rate
gamma         = 0.99     # discount factor
epsilon       = 1.0      # initial exploration rate
epsilon_min   = 0.05     # minimum exploration rate
epsilon_decay = 0.995    # exponential decay rate
render        = False    # render DURING training (slow — set True to watch)
fps_training  = 6        # FPS if render=True during training

# Live-game settings
fps_live      = 8        # starting FPS for the live game window

Q_TABLE_PATH  = os.path.join(RESULTS_DIR, 'q_table.npy')

# Helper for greedy rollouts used in analysis and GIF export.
def run_greedy_episode(q_table, env=None, close_after=True, max_steps=300):
    """Run one episode using the greedy policy (no exploration)."""
    if env is None:
        env = create_env(render_mode=False)
    obs, _ = env.reset()
    path   = [(int(obs[0]), int(obs[1]), int(obs[2]))]
    total_r = 0
    info = {}
    for _ in range(max_steps):
        r, c, carrying = int(obs[0]), int(obs[1]), int(obs[2])
        action = np.argmax(q_table[r, c, carrying])
        obs, reward, done, info = env.step(action)
        path.append((int(obs[0]), int(obs[1]), int(obs[2])))
        total_r += reward
        if done:
            break
    if close_after:
        env.close()
    return path, total_r, info.get('event', '?')


# Live-game viewer with adjustable speed.
def run_live_game(q_table, fps=8):
    """
    Watch the trained agent play one episode.
    Window stays open until you close it manually.
    Controls: +/- speed,  Q or ESC or close window to exit.
    """
    import pygame

    env        = create_env(render_mode=True)
    obs, _     = env.reset()
    total_r    = 0.0
    done       = False
    step_count = 0
    font_big   = pygame.font.SysFont('Arial', 18, bold=True)
    clock      = pygame.time.Clock()

    print("\n  Live game open. Controls: +/- speed   Q quit   close window to continue")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close()
                    return
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                 pygame.K_KP_PLUS):
                    fps = min(30, fps + 2)
                if event.key == pygame.K_MINUS:
                    fps = max(1, fps - 2)

        if done:
            obs, _ = env.reset()
            total_r = 0.0
            step_count = 0
            done = False
            outcome = 'reset'

        r, c, carrying = int(obs[0]), int(obs[1]), int(obs[2])
        action         = np.argmax(q_table[r, c, carrying])
        obs, reward, done, info = env.step(action)
        total_r    += reward
        step_count += 1

        extra = (f"Steps:{step_count}  R:{total_r:.0f}  "
                 f"FPS:{fps}  [+/-] speed  [Q] quit")
        env.render(fps=fps, extra_info=extra)


# Save a greedy episode as an animated GIF.
def save_gameplay_gif(q_table, filename=None):
    if filename is None:
        filename = os.path.join(RESULTS_DIR, 'final_game.gif')

    import pygame
    from PIL import Image as PILImage

    env   = create_env(render_mode=True)
    obs, _ = env.reset()
    frames = []
    done   = False

    for _ in range(300):
        r, c, carrying = int(obs[0]), int(obs[1]), int(obs[2])
        action = np.argmax(q_table[r, c, carrying])
        obs, reward, done, info = env.step(action)
        env.render(fps=20, extra_info='Recording GIF...')
        # capture pygame surface
        raw   = pygame.surfarray.array3d(env.screen)
        frame = PILImage.fromarray(raw.transpose(1, 0, 2))
        frames.append(frame.resize((GRID_SIZE*CELL_SIZE//2,
                                    (GRID_SIZE*CELL_SIZE+60)//2),
                                   PILImage.LANCZOS))
        if done:
            break

    env.close()
    if frames:
        frames[0].save(filename, save_all=True,
                       append_images=frames[1:],
                       duration=100, loop=0)
        print(f"  -> saved {filename}  ({len(frames)} frames)")
    else:
        print("  No frames captured for GIF")


# Show the greedy path at several training checkpoints.
def save_training_progress(checkpoints, q_tables, outcomes):
    """Draw the museum with the greedy route at several training checkpoints."""
    n = len(checkpoints)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5))
    if n == 1: axes = [axes]

    for ax, ep, q, (path, reward, event) in zip(
            axes, checkpoints, q_tables, outcomes):
        # draw grid colours
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                from Q_learning import CELL_COLORS
                col = CELL_COLORS.get(GRID[rr][cc], '#f7f7f7')
                ax.add_patch(plt.Rectangle((cc, GRID_SIZE-1-rr),
                             1, 1, facecolor=col, edgecolor='white', lw=1))
                sym = GRID[rr][cc]
                if sym not in ('.', 'S'):
                    ax.text(cc+0.5, GRID_SIZE-0.5-rr, sym,
                            ha='center', va='center', fontsize=8,
                            color='white' if sym in ('R','A') else 'black',
                            fontweight='bold')
        # draw path
        xs = [cc+0.5 for (_,cc,_) in path]
        ys = [GRID_SIZE-0.5-rr for (rr,_,_) in path]
        ax.plot(xs, ys, '-', color='#08306b', lw=2, alpha=0.7, zorder=3)
        ax.plot(xs[0], ys[0], 'o', color='#2171b5', ms=10, zorder=4)
        ax.plot(xs[-1], ys[-1], '*', color='#238b45', ms=14, zorder=4)
        # Mark the moment the diamond is picked up.
        for i in range(1, len(path)):
            if path[i-1][2] == 0 and path[i][2] == 1:  # carrying just flipped
                dx = path[i][1] + 0.5
                dy = GRID_SIZE - 0.5 - path[i][0]
                ax.plot(dx, dy, 'D', color='#ffd700', ms=10, zorder=5,
                        markeredgecolor='black', markeredgewidth=1)
                break
        title_ep = 'untrained' if ep == 0 else f'after {ep} episodes'
        outcome_label = {
            'escaped_win': 'WIN',
            'caught':      'caught',
            'exit_empty':  'exit without diamond',
            'step':        'timed out (still learning)',
        }.get(event, event)
        ax.set_title(f'{title_ep}\n{outcome_label} | R={reward:.0f}', fontsize=10)
        ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal'); ax.axis('off')

    fig.suptitle('Training progress — greedy route at checkpoints', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(RESULTS_DIR, 'training_progress.png')
    fig.savefig(fname, dpi=110, bbox_inches='tight')
    # Display the figure after saving it.
    plt.show()
    plt.close(fig)
    print(f"  -> saved {fname}")


# Create a README describing the generated outputs.
def write_results_readme():
    lines = [
        "MUSEUM THIEF — RESULTS FOLDER",
        "PADM Project A, THI Ingolstadt SS26",
        "=" * 50,
        "",
        "q_table.npy               — trained Q-table (11x11x2x4 = 968 values)",
        "training_curves.png       — reward, win-rate, epsilon over episodes",
        "training_progress.png     — greedy route at training checkpoints",
        "coverage_heatmap.png      — how many times each cell was visited",
        "",
        "exp1_exploration_strategies.png — Exp1: exponential vs linear epsilon-decay",
        "exp1_explanation.txt            — theory justification for Exp1",
        "exp2_learning_rates.png         — Exp2: alpha=0.01 vs 0.1 vs 0.3",
        "exp2_explanation.txt            — theory justification for Exp2",
        "",
        "qtable_actions_phase0.png — Q-values per action, not carrying",
        "qtable_actions_phase1.png — Q-values per action, carrying diamond",
        "policy_phase0.png         — optimal policy arrows, not carrying",
        "policy_phase1.png         — optimal policy arrows, carrying diamond",
        "",
        "final_game.gif            — animated GIF of trained agent winning",
        "",
        "GRID LAYOUT (11x11 — rows 0-10, cols 0-10):",
        "  S=Start      D=Diamond(+50 main goal)",
        "  E=Exit       (+100 if carrying / -10 if not)",
        "  G=Gold       (+5 one-time bonus)",
        "  T=Teleport   (2,8) <-> (8,2)  two-way pair",
        "  C=Camera     (-3 penalty, walk through)",
        "  R=Guard      (-20 caught, episode ends)",
        "  A=Alarm      (-15 caught, episode ends)",
        "",
        "POSITIONS (row, col):",
        "  START   (0,0)    EXIT    (0,5)    DIAMOND (10,10)",
        "  GOLD    (0,10)   (5,0)   (5,10)   (10,5)",
        "  GUARD   (1,1)    (1,9)   (5,5)    (9,1)   (9,9)",
        "  ALARM   (2,5)    (5,2)   (5,8)    (8,5)",
        "  CAMERA  (3,3)    (3,7)   (7,3)    (7,7)",
        "  TELEPORT (2,8) <-> (8,2)",
        "",
        "TWO-STAGE TASK:",
        "  Phase 0 (carrying=False): Start(0,0) -> Diamond(10,10)",
        "  Phase 1 (carrying=True):  Diamond(10,10) -> Exit(0,5)",
        "  Reset each episode: (0,0), carrying=False",
        "  Q-table shape: 11x11x2x4 = 968 values",
    ]
    with open(os.path.join(RESULTS_DIR, 'README.txt'), 'w', encoding="utf-8") as f:
        f.write('\n'.join(lines))
    print("  -> saved results/README.txt")


def show_saved_results():
    """Open and display all previously saved result images on screen."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    images_to_show = [
        ('training_curves.png',              'Training Curves'),
        ('training_progress.png',            'Training Progress'),
        ('coverage_heatmap.png',             'Exploration Coverage'),
        ('exp1_exploration_strategies.png',  'Experiment 1 — Exploration Strategies'),
        ('exp2_learning_rates.png',          'Experiment 2 — Learning Rates'),
    ]

    for filename, title in images_to_show:
        fpath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(fpath):
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.imshow(mpimg.imread(fpath))
            ax.axis('off')
            fig.suptitle(title, fontsize=13)
            fig.tight_layout()
            plt.show()
            plt.close(fig)
            print(f"  -> showed {filename}")
        else:
            print(f"  -> {filename} not found in results/ (run --train first)")


# Main entry point
if __name__ == '__main__':

    print("=" * 60)
    print("  MUSEUM THIEF — PADM Project A")
    print("=" * 60)

    if not _TRAIN:
        print("\n  Showing saved results from last training run...")
        show_saved_results()

    print("\n[0/5] Saving environment layout preview...")
    save_environment_preview()

    q_table = None
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Training
    # ------------------------------------------------------------------
    if RUN_TRAINING:
        print("\n[1/5] Training the Museum Thief agent...")
        env = create_env(render_mode=render)
        # === MUSEUM CHANGE: capture history for curves + progress strip ===
        checkpoints = [0, 100, 500, 1000, 3000, 5000]
        snap_q_tables = []
        snap_outcomes = []

        q_table, history = train_q_learning(
            env=env,
            no_episodes=no_episodes,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            alpha=alpha,
            gamma=gamma,
            render=render,
            fps=fps_training,
            q_table_save_path=Q_TABLE_PATH,
            return_history=True,
            snapshot_checkpoints=checkpoints,
            snapshot_store=snap_q_tables,
        )
        if render:
            env.close()

        print("\n  Saving training curves...")
        plot_training_curves(history)

        print("  Saving coverage heatmap...")
        plot_coverage(history['visit_counts'])

        # Training progress strip at checkpoints
        print("  Saving training progress strip...")
        for q_snapshot in snap_q_tables:
            path, reward, event = run_greedy_episode(
                q_snapshot, create_env(render_mode=False), max_steps=300)
            snap_outcomes.append((path, reward, event))

        save_training_progress(checkpoints, snap_q_tables, snap_outcomes)

    # ------------------------------------------------------------------
    # 2. Experiments
    # ------------------------------------------------------------------
    if RUN_EXPERIMENTS:
        print("\n[2/5] Running experiments...")
        experiment_exploration(
            env_factory=create_env,
            no_episodes=no_episodes,
            alpha=alpha, gamma=gamma)
        experiment_learning_rates(
            env_factory=create_env,
            no_episodes=no_episodes,
            epsilon_decay=epsilon_decay, gamma=gamma)

    # ------------------------------------------------------------------
    # 3. Analysis — Q-table visualizations
    # ------------------------------------------------------------------
    if RUN_ANALYSIS:
        print("\n[3/5] Generating Q-table analysis...")
        if q_table is None and os.path.exists(Q_TABLE_PATH):
            q_table = np.load(Q_TABLE_PATH)
        if q_table is not None:
            visualize_q_table(Q_TABLE_PATH)
        else:
            print("  No Q-table found — skipping analysis. Run training first.")

    # ------------------------------------------------------------------
    # 4. Live game — watch the trained agent
    # ------------------------------------------------------------------
    if RUN_LIVE_GAME:
        print("\n[4/5] Starting live game...")
        if q_table is None and os.path.exists(Q_TABLE_PATH):
            q_table = np.load(Q_TABLE_PATH)
        if q_table is not None:
            run_live_game(q_table, fps=fps_live)
        else:
            print("  No Q-table found — skipping live game.")

    # ------------------------------------------------------------------
    # 5. Save gameplay GIF
    # ------------------------------------------------------------------
    if SAVE_GIF:
        print("\n[5/5] Saving gameplay GIF...")
        if q_table is None and os.path.exists(Q_TABLE_PATH):
            q_table = np.load(Q_TABLE_PATH)
        if q_table is not None:
            save_gameplay_gif(q_table)
        else:
            print("  No Q-table — skipping GIF.")

    # ------------------------------------------------------------------
    # Write results README
    # ------------------------------------------------------------------
    write_results_readme()

    print("\n" + "=" * 60)
    print("  ALL DONE. Check the results/ folder.")
    print("=" * 60)
