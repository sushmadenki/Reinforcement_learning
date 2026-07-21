MUSEUM THIEF — RESULTS FOLDER
PADM Project A, THI Ingolstadt SS26
==================================================

q_table.npy               — trained Q-table (11x11x2x4 = 968 values)
training_curves.png       — reward, win-rate, epsilon over episodes
training_progress.png     — greedy route at training checkpoints
coverage_heatmap.png      — how many times each cell was visited

exp1_exploration_strategies.png — Exp1: exponential vs linear epsilon-decay
exp1_explanation.txt            — theory justification for Exp1
exp2_learning_rates.png         — Exp2: alpha=0.01 vs 0.1 vs 0.3
exp2_explanation.txt            — theory justification for Exp2

qtable_actions_phase0.png — Q-values per action, not carrying
qtable_actions_phase1.png — Q-values per action, carrying diamond
policy_phase0.png         — optimal policy arrows, not carrying
policy_phase1.png         — optimal policy arrows, carrying diamond

final_game.gif            — animated GIF of trained agent winning

GRID LAYOUT (11x11 — rows 0-10, cols 0-10):
  S=Start      D=Diamond(+50 main goal)
  E=Exit       (+100 if carrying / -10 if not)
  G=Gold       (+5 one-time bonus)
  T=Teleport   (2,8) <-> (8,2)  two-way pair
  C=Camera     (-3 penalty, walk through)
  R=Guard      (-20 caught, episode ends)
  A=Alarm      (-15 caught, episode ends)

POSITIONS (row, col):
  START   (0,0)    EXIT    (0,5)    DIAMOND (10,10)
  GOLD    (0,10)   (5,0)   (5,10)   (10,5)
  GUARD   (1,1)    (1,9)   (5,5)    (9,1)   (9,9)
  ALARM   (2,5)    (5,2)   (5,8)    (8,5)
  CAMERA  (3,3)    (3,7)   (7,3)    (7,7)
  TELEPORT (2,8) <-> (8,2)

TWO-STAGE TASK:
  Phase 0 (carrying=False): Start(0,0) -> Diamond(10,10)
  Phase 1 (carrying=True):  Diamond(10,10) -> Exit(0,5)
  Reset each episode: (0,0), carrying=False
  Q-table shape: 11x11x2x4 = 968 values
