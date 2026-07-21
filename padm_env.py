# padm_env.py
# Museum environment for the museum-heist RL task.
# Implements the two-stage reward structure and rendering logic.

"""
Museum Thief Environment (padm_env.py)
=======================================
Custom 11x11 grid environment for the Museum Thief Q-learning project.

The thief must complete a TWO-STAGE task:
    Phase 0: Navigate from Start (0,0) to Diamond (10,10) — picks up +50
    Phase 1: Navigate from Diamond (10,10) to Exit (0,5)  — escapes for +100

STATE:  (row, col, carrying)  — 11 x 11 x 2 = 242 states
ACTION: Up(0), Down(1), Right(2), Left(3)
Q-TABLE shape: (11, 11, 2, 4) = 968 values

USAGE:
    from padm_env import create_env
    env = create_env(render_mode=True)   # render_mode=False for headless training
    obs, info = env.reset()
    obs, reward, done, info = env.step(action)
    env.render(fps=10)
    env.close()
"""


# Imports
import sys
import os
import numpy as np

# Import gymnasium, with a lightweight fallback if it is unavailable.
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym  # pyright: ignore[reportMissingImports]
    except ImportError:
        # Minimal stub — only what the template actually calls
        class _Discrete:
            def __init__(self, n): self.n = n
            def sample(self): return int(np.random.randint(self.n))
        class _Box:
            def __init__(self, low, high, shape, dtype): pass
        class _Spaces:
            Discrete = _Discrete
            Box      = _Box
        class _GymStub:
            spaces = _Spaces()
            class Env:
                pass
        gym = _GymStub()

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    # Minimal pygame stub so the module imports cleanly even without display
    class _FakeSurface:
        def blit(self, *a, **k): pass
        def fill(self, *a, **k): pass
    class _FakeDisplay:
        def set_caption(self, *a): pass
        def set_mode(self, *a): return _FakeSurface()
        def flip(self): pass
    class _FakeFont:
        def render(self, *a, **k): return _FakeSurface()
    class _FakeSysFont:
        def __init__(self, *a, **k): pass
        def render(self, *a, **k): return _FakeSurface()
    class _FakeClock:
        def tick(self, *a): pass
    class _FakeImage:
        def load(self, p):
            s = _FakeSurface()
            s.convert_alpha = lambda: s
            return s
    class _FakeDraw:
        def rect(self, *a, **k): pass
        def circle(self, *a, **k): pass
    class _FakeSurfaceAlpha(_FakeSurface):
        def __init__(self, size, flag=0): pass
    class _FakeSurfarray:
        def array3d(self, s): import numpy as np; return np.zeros((100,100,3),dtype=np.uint8)
    class _FakePygame:
        QUIT    = 0; KEYDOWN = 1; SRCALPHA = 0
        K_q = K_ESCAPE = K_PLUS = K_EQUALS = K_KP_PLUS = K_MINUS = K_r = K_SPACE = 0
        display  = _FakeDisplay()
        font     = type('F', (), {'SysFont': lambda *a,**k: _FakeSysFont()})()
        image    = _FakeImage()
        draw     = _FakeDraw()
        surfarray = _FakeSurfarray()
        Surface  = _FakeSurfaceAlpha
        def init(self): pass
        def quit(self): pass
        def event_get(self): return []
        class event:
            @staticmethod
            def get(): return []
        class time:
            class Clock:
                def tick(self, *a): pass
            @staticmethod
            def wait(ms): pass
    pygame = _FakePygame()


# Grid layout constants for the 11x11 museum map.
GRID_SIZE  = 11
CELL_SIZE  = 75          # pixels per cell
WIN_W      = GRID_SIZE * CELL_SIZE
WIN_H      = GRID_SIZE * CELL_SIZE + 60   # +60 for status bar

# Museum layout and symbol meanings.
#
#     SYMBOL KEY:
#       S  = Start         → always reset here every episode
#       D  = Diamond       → main goal (+50), sets carrying=True
#       E  = Exit          → secondary goal (+100 if carrying, -10 if not)
#       G  = Gold          → one-time bonus (+5)
#       T  = Teleport      → steps on (2,8) send to (8,2) and vice versa
#       C  = Camera        → penalty (-3) every visit, walk through
#       R  = Guard         → caught (-20), episode ends
#       A  = Alarm         → caught (-15), episode ends
#       .  = empty floor
#
#     POSITIONS (row, col):
#       S(0,0)   E(0,5)   G(0,10)
#       R(1,1)   R(1,9)
#       T(2,2)   A(2,5)   T(2,8)
#       C(3,3)   C(3,7)
#       G(5,0)   A(5,2)   R(5,5)   A(5,8)   G(5,10)
#       C(7,3)   C(7,7)
#       T(8,2)   A(8,5)   T(8,8)
#       R(9,1)   R(9,9)
#       G(10,5)  D(10,10)
# ===
GRID = [
    ['S', '.', '.', '.', '.', 'E', '.', '.', '.', '.', 'G'],  # row 0
    ['.', 'R', '.', '.', '.', '.', '.', '.', '.', 'R', '.'],  # row 1
    ['.', '.', '.', '.', '.', 'A', '.', '.', 'T', '.', '.'],  # row 2
    ['.', '.', '.', 'C', '.', '.', '.', 'C', '.', '.', '.'],  # row 3
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # row 4
    ['G', '.', 'A', '.', '.', 'R', '.', '.', 'A', '.', 'G'],  # row 5
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # row 6
    ['.', '.', '.', 'C', '.', '.', '.', 'C', '.', '.', '.'],  # row 7
    ['.', '.', 'T', '.', '.', 'A', '.', '.', '.', '.', '.'],  # row 8
    ['.', 'R', '.', '.', '.', '.', '.', '.', '.', 'R', '.'],  # row 9
    ['.', '.', '.', '.', '.', 'G', '.', '.', '.', '.', 'D'],  # row 10
]
# Teleport pairs for the museum map.
TELEPORT_PAIR = {
    (2, 8): (8, 2),   # V: top-right  → bottom-left
    (8, 2): (2, 8),   # V: bottom-left → top-right
}

# Reward values for each map cell.
REWARDS = {
    'D':  50,          # Diamond pickup → main goal
    'E':  100,         # Exit while carrying → WIN
    'G':  5,           # Gold bonus (one-time)
    'R': -20,          # Guard → caught, episode ends
    'A': -15,          # Alarm → caught, episode ends
    'C': -3,           # Camera penalty (every visit)
    'T':  0,           # Teleport T (no reward, just moves you)
    '.':  0,           # Empty floor
    'S':  0,           # Start
    'E_empty': -10,    # Exit without diamond → fail
}
STEP_COST = -1   # every step costs 1 to encourage short routes


# Asset names used for each symbol.
CELL_ASSET = {
    'S': 'floor2',    'D': 'diamond',   'E': 'exit',
    'G': 'gold',      'T': 'teleport',  'C': 'camera',
    'R': 'guard',     'A': 'alarm',     '.': 'floor',
}

# Human-readable names for the grid symbols.
CELL_NAME = {
    'S': 'Start',    'D': 'Diamond',   'E': 'Exit',
    'G': 'Gold',     'T': 'Teleport',  'C': 'Camera',
    'R': 'Guard',    'A': 'Alarm',     '.': 'Floor',
}


# Load original uploaded assets and resize them when drawing.

# Map asset name → original uploaded filename
ORIGINAL_FILES = {
    'thief':    'thief.gif',
    'guard':    'guard.gif',
    'alarm':    'alarm.gif',
    'camera':   'camera.gif',
    'diamond':  'diamonds.gif',
    'gold':     'gold.gif',
    'teleport': 'teleport.gif',
    'exit':     'exit.jpg',
    'floor':    'floor.jpg',
    'floor2':   'floor2.jpg',
}

# Where your original uploaded files are stored
UPLOADS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'uploads')


def _load_frames(name):
    """Load ALL frames from the original uploaded file.
    GIFs keep every frame. JPGs/PNGs load as a single frame.
    Each frame is resized to CELL_SIZE at load time so drawing is fast."""
    from PIL import Image as PILImage

    fname = ORIGINAL_FILES.get(name)
    if fname is None:
        return None

    # Load from uploads/ folder
    path = os.path.join(UPLOADS_DIR, fname)

    if not os.path.exists(path):
        return None

    try:
        pil_img = PILImage.open(path)
        frames  = []
        n       = getattr(pil_img, 'n_frames', 1)

        for i in range(n):
            try:
                pil_img.seek(i)
                frame = pil_img.convert('RGBA').resize(
                    (CELL_SIZE, CELL_SIZE), PILImage.LANCZOS)
                # convert PIL frame → pygame surface
                raw   = frame.tobytes()
                surf  = pygame.image.fromstring(
                    raw, (CELL_SIZE, CELL_SIZE), 'RGBA')
                frames.append(surf.convert_alpha())
            except EOFError:
                break

        return frames if frames else None

    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


# Museum environment class.
class MuseumEnv(gym.Env):

    def __init__(self, render_mode=True) -> None:
        super(MuseumEnv, self).__init__()

        # --- same pattern as template ---
        self.grid_size  = GRID_SIZE
        self.cell_size  = CELL_SIZE
        self.state      = None
        self.reward     = 0
        self.info       = {}
        self.done       = False

        # Two-stage task state.
        self.carrying   = False          # True once diamond is picked up
        self.collected  = set()          # one-time cells already triggered

        # Action and observation spaces.
        self.action_space      = gym.spaces.Discrete(4)   # 0=Up 1=Down 2=Right 3=Left
        self.observation_space = gym.spaces.Box(
            low=0, high=GRID_SIZE - 1, shape=(3,), dtype=np.int32)
        # obs = [row, col, carrying]  ← +1 dim vs template for carrying flag

        # --- pygame init ---
        self.render_mode = render_mode
        if render_mode and PYGAME_AVAILABLE:
            pygame.init()
            pygame.display.set_caption("Museum Thief — PADM Project A")
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            self.font   = pygame.font.SysFont('Arial', 16, bold=True)
            self.clock  = pygame.time.Clock()
            # Load sprite frames directly from the uploads folder.
            self._surfaces = {}
            for sym, name in CELL_ASSET.items():
                if name not in self._surfaces:
                    frames = _load_frames(name)
                    if frames:
                        self._surfaces[name] = frames
                    else:
                        print(f"  Warning: could not load asset '{name}'")
            self._thief_frames = self._surfaces.get('thief') or []
            self._anim_tick    = 0   # global animation counter

    # ------------------------------------------------------------------
    # Method 1: reset()  — same signature as template
    # ------------------------------------------------------------------
    def reset(self):
        # Reset the episode state.
        self.state    = np.array([0, 0])   # always start at (0,0)
        self.carrying = False
        self.collected = set()
        self.done     = False
        self.reward   = 0
        self.info['event'] = 'reset'
        return self._obs(), self.info

    # ------------------------------------------------------------------
    # Method 2: step()  — same structure as template
    # ------------------------------------------------------------------
    def step(self, action):
        # Apply the chosen action.
        r, c = int(self.state[0]), int(self.state[1])

        # Move (with boundary check — same as template):
        if action == 0 and r > 0:               r -= 1   # Up
        if action == 1 and r < GRID_SIZE - 1:   r += 1   # Down
        if action == 2 and c < GRID_SIZE - 1:   c += 1   # Right
        if action == 3 and c > 0:               c -= 1   # Left

        # Apply teleportation if needed.
        if (r, c) in TELEPORT_PAIR:
            r, c = TELEPORT_PAIR[(r, c)]

        self.state = np.array([r, c])
        cell = GRID[r][c]

        # Apply the reward logic for the current cell.
        self.reward = STEP_COST
        self.done   = False
        event       = 'step'

        if cell == 'D' and not self.carrying:
            # Primary goal: pick up diamond
            self.reward  += REWARDS['D']
            self.carrying = True
            event         = 'diamond_collected'

        elif cell == 'E':
            # Secondary goal: exit
            if self.carrying:
                self.reward += REWARDS['E']
                self.done    = True
                event        = 'escaped_win'
            else:
                self.reward += REWARDS['E_empty']
                self.done    = True
                event        = 'exit_empty'

        elif cell in ('R', 'A'):
            # Episode-ending danger cells
            self.reward += REWARDS[cell]
            self.done    = True
            event        = 'caught'

        elif cell in 'G':
            # One-time bonus/trap cells
            if (r, c) not in self.collected:
                self.reward += REWARDS['G']
                self.collected.add((r, c))

        # === MUSEUM CHANGE: only C (camera) is a walk-through penalty
        #     removed L (laser) not in new grid ===
        elif cell == 'C':
            # Camera penalty — every visit, walk through
            self.reward += REWARDS['C']

        self.info['event']    = event
        self.info['carrying'] = self.carrying
        self.info['cell']     = cell

        return self._obs(), self.reward, self.done, self.info

    # ------------------------------------------------------------------
    # Method 3: render() — renders animated images
    # ------------------------------------------------------------------
    def render(self, fps=10, extra_info=None):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self._anim_tick += 1

        # Draw grid cells with animated GIF frames:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                sym  = GRID[row][col]
                name = CELL_ASSET[sym]
                # === MUSEUM CHANGE: Draw pre-processed asset frames ===
                frames = self._surfaces.get(name)
                if frames:
                    n_frames = len(frames)
                    frame_idx = (self._anim_tick // 3) % n_frames
                    self.screen.blit(frames[frame_idx],
                                     (col * CELL_SIZE, row * CELL_SIZE))
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200),
                                     (col*CELL_SIZE, row*CELL_SIZE,
                                      CELL_SIZE, CELL_SIZE))

                # Grid border
                pygame.draw.rect(self.screen, (80, 80, 80),
                                 (col*CELL_SIZE, row*CELL_SIZE,
                                  CELL_SIZE, CELL_SIZE), 1)

        # Draw collected one-time cells as dimmed overlay:
        for (cr, cc) in self.collected:
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((0, 0, 0, 100))
            self.screen.blit(s, (cc*CELL_SIZE, cr*CELL_SIZE))

        # Draw thief (animated):
        r, c = int(self.state[0]), int(self.state[1])
        if self._thief_frames:
            tf_idx = (self._anim_tick // 2) % len(self._thief_frames)
            self.screen.blit(self._thief_frames[tf_idx],
                             (c*CELL_SIZE, r*CELL_SIZE))
        else:
            pygame.draw.circle(self.screen, (0, 100, 255),
                                (c*CELL_SIZE + CELL_SIZE//2,
                                 r*CELL_SIZE + CELL_SIZE//2), 20)

        # === MUSEUM CHANGE: Diamond badge on thief when carrying ===
        if self.carrying:
            badge = self.font.render('💎', True, (255, 215, 0))
            self.screen.blit(badge, (c*CELL_SIZE + 2, r*CELL_SIZE + 2))

        # Status bar at bottom:
        bar_y = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(self.screen, (20, 20, 40),
                         (0, bar_y, WIN_W, 60))
        carry_txt = '💎 CARRYING' if self.carrying else 'searching...'
        cell_name = CELL_NAME.get(GRID[r][c], '?')
        status = f"Pos:({r},{c}) Cell:{cell_name}  {carry_txt}"
        if extra_info:
            status += f"  |  {extra_info}"
        self.screen.blit(
            self.font.render(status, True, (220, 220, 220)), (8, bar_y + 8))

        pygame.display.flip()
        self.clock.tick(fps)

    # ------------------------------------------------------------------
    # Method 4: close()  — same as template
    # ------------------------------------------------------------------
    def close(self):
        if self.render_mode:
            pygame.quit()

    # ------------------------------------------------------------------
    # Helper: observation vector  [row, col, carrying]
    # ------------------------------------------------------------------
    def _obs(self):
        # === MUSEUM CHANGE: obs includes carrying flag (index 2) ===
        return np.array([self.state[0], self.state[1],
                         int(self.carrying)], dtype=np.int32)


# === MUSEUM CHANGE: create_env factory — same pattern as template ===
def create_env(render_mode=True):
    return MuseumEnv(render_mode=render_mode)



# === MUSEUM CHANGE: Saves a static preview of the museum layout to
#     results/environment_map.png — bare grid (1px hairline gaps, no
#     numbers/labels/rewards on cells) + legend + two-stage task box.
#     Uses frame 0 of each asset (a PNG can only show one frame; the GIFs
#     still animate fully in render() / the live game / the gameplay GIF). ===
def save_environment_preview(filename=None):
    from PIL import Image, ImageDraw, ImageFont
    import os as _os

    if filename is None:
        filename = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            'results', 'environment_map.png')
    _os.makedirs(_os.path.dirname(filename), exist_ok=True)

    GAP = 1
    grid_w = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GAP
    grid_h = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GAP

    legend_names = {
        'S': 'Start',
        'D': 'Diamond — goal 1 (+50)',
        'E': 'Exit — goal 2 (+100)',
        'G': 'Gold (+5 one-time)',
        'T': 'Teleport  (2,8)↔(8,2)',
        'C': 'Camera (-3 penalty)',
        'R': 'Guard (-20 caught)',
        'A': 'Alarm (-15 caught)',
    }
    legend_order = ['S', 'D', 'E', 'G', 'T', 'C', 'R', 'A']
    swatch, row_h, legend_w = 28, 36, 260
    legend_top_h = 28 + len(legend_order) * row_h + 16

    # === MUSEUM CHANGE: updated positions for 11x11 grid ===
    box_lines = [
        "Two-stage task",
        "Phase 0: Start (0,0) -> Diamond (10,10)",
        "Phase 1: Diamond (10,10) -> Exit (0,5)",
        "",
        "State = (row, col, carrying)",
        "Episode always resets to (0,0)",
    ]
    box_h = 18 + len(box_lines) * 20 + 16

    H = max(grid_h, legend_top_h + box_h)
    W = grid_w + 20 + legend_w
    canvas = Image.new('RGB', (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)

# === MUSEUM CHANGE: load from uploads/ directly — no assets/ folder needed ===
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            sym        = GRID[r][c]
            asset_name = CELL_ASSET[sym]
            fname      = ORIGINAL_FILES.get(asset_name)
            img_path   = _os.path.join(UPLOADS_DIR, fname) if fname else None
            if img_path and _os.path.exists(img_path):
                cell_img = Image.open(img_path).convert('RGB').resize(
                    (CELL_SIZE, CELL_SIZE), Image.LANCZOS)
            else:
                # fallback: plain colored square if file not found
                cell_img = Image.new('RGB', (CELL_SIZE, CELL_SIZE), (200,200,200))
            canvas.paste(cell_img, (c*(CELL_SIZE+GAP), r*(CELL_SIZE+GAP)))

    try:
        f_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        f_body = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        f_box_t = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        f_box_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        f_title = f_body = f_box_t = f_box_b = ImageFont.load_default()

    lx, ly = grid_w + 20, 10
    draw.text((lx, ly), "Legend", font=f_title, fill=(20, 20, 20))
    ly += 28
# === MUSEUM CHANGE: load legend swatches from uploads/ ===
    for sym in legend_order:
        asset_name = CELL_ASSET[sym]
        fname      = ORIGINAL_FILES.get(asset_name)
        img_path   = _os.path.join(UPLOADS_DIR, fname) if fname else None
        if img_path and _os.path.exists(img_path):
            sw_img = Image.open(img_path).convert('RGB').resize(
                (swatch, swatch), Image.LANCZOS)
        else:
            sw_img = Image.new('RGB', (swatch, swatch), (200,200,200))
        canvas.paste(sw_img, (lx, ly))
        draw.text((lx+swatch+8, ly+5), legend_names[sym],
                 font=f_body, fill=(40, 40, 40))
        ly += row_h

    ly += 16
    box_x0, box_y0 = lx, ly
    box_x1, box_y1 = lx + legend_w - 10, ly + box_h
    draw.rectangle([box_x0, box_y0, box_x1, box_y1],
                   outline=(180, 180, 175), width=1)
    ty = box_y0 + 10
    draw.text((box_x0+12, ty), box_lines[0], font=f_box_t, fill=(20, 20, 20))
    ty += 22
    for line in box_lines[1:]:
        draw.text((box_x0+12, ty), line, font=f_box_b, fill=(60, 60, 60))
        ty += 20

    canvas.save(filename)
    print(f"  -> saved {filename}")