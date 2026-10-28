"""
❄️ Frozen Lake RL Puzzle Solver — Streamlit App
================================================
Run with:  streamlit run frozen_lake_rl.py

Requirements:
    pip install streamlit numpy matplotlib seaborn
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Frozen Lake RL Solver",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0f4ff;
}
h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem;
    background: linear-gradient(90deg, #00d4ff, #0099cc, #00ffcc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; color: #7dd4f8; }
.metric-card {
    background: rgba(0,180,255,0.07); border: 1px solid rgba(0,180,255,0.2);
    border-radius: 12px; padding: 16px 20px; text-align: center; margin-bottom: 12px;
}
.metric-value { font-family: 'Space Mono',monospace; font-size: 2rem; font-weight: 700; color: #00d4ff; }
.metric-label { font-size: 0.75rem; color: #8ab4d4; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
div[data-testid="stSidebar"] { background: rgba(10,20,40,0.95); border-right: 1px solid rgba(0,180,255,0.15); }
div[data-testid="stSidebar"] label { color: #8ab4d4 !important; font-size:0.82rem; text-transform:uppercase; letter-spacing:1px; }
.stButton > button {
    background: linear-gradient(135deg,#0066aa,#004477); color: #00d4ff;
    border: 1px solid #00d4ff55; border-radius: 8px;
    font-family: 'Space Mono',monospace; font-weight:700; font-size:0.85rem;
    letter-spacing:1px; padding:10px 20px; width:100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#0099ff,#0066aa); border-color:#00d4ff;
    transform:translateY(-1px); box-shadow:0 4px 20px rgba(0,180,255,0.25);
}
.stProgress > div > div { background: linear-gradient(90deg,#00d4ff,#00ffcc); border-radius:4px; }
[data-testid="stMetricValue"] { font-family:'Space Mono',monospace; color:#00d4ff; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Environment definition
# ─────────────────────────────────────────────
MAPS = {
    "4x4 Classic": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "4x4 Tricky": [
        "SFFF",
        "HFHF",
        "FFHF",
        "FHFG",
    ],
    "8x8 Large": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

ACTIONS   = {0: "← Left", 1: "↓ Down", 2: "→ Right", 3: "↑ Up"}
ACT_SYM   = {0: "←", 1: "↓", 2: "→", 3: "↑"}
ACT_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

TILE_EMOJI  = {"S": "🟦", "F": "⬜", "H": "🕳️", "G": "🏆"}
TILE_COLOR  = {"S": "#1a6fa8", "F": "#cce8f4", "H": "#181828", "G": "#ffd700"}
TILE_BORDER = {"S": "#00d4ff", "F": "#90cce8", "H": "#4a4a7a", "G": "#ffaa00"}


class FrozenLakeEnv:
    def __init__(self, map_name="4x4 Classic", is_slippery=True):
        self.desc       = MAPS[map_name]
        self.nrow       = len(self.desc)
        self.ncol       = len(self.desc[0])
        self.n_states   = self.nrow * self.ncol
        self.n_actions  = 4
        self.is_slippery = is_slippery
        for r, row in enumerate(self.desc):
            for c, ch in enumerate(row):
                if ch == "S": self.start = (r, c)
                if ch == "G": self.goal  = (r, c)
        self.pos = self.start

    def reset(self):
        self.pos = self.start
        return self._state()

    def _state(self):
        return self.pos[0] * self.ncol + self.pos[1]

    def step(self, intended_action):
        if self.is_slippery:
            actual = random.choice([
                (intended_action - 1) % 4,
                intended_action,
                (intended_action + 1) % 4,
            ])
        else:
            actual = intended_action

        dr, dc = ACT_DELTA[actual]
        nr = max(0, min(self.nrow - 1, self.pos[0] + dr))
        nc = max(0, min(self.ncol - 1, self.pos[1] + dc))
        self.pos = (nr, nc)
        tile = self.desc[nr][nc]
        done = tile in ("H", "G")
        reward = 1.0 if tile == "G" else (-1.0 if tile == "H" else -0.01)
        return self._state(), reward, done, tile, actual

    def pos_from_state(self, s):
        return (s // self.ncol, s % self.ncol)


# ─────────────────────────────────────────────
# Q-Learning
# ─────────────────────────────────────────────
def train_q_learning(env, episodes, alpha, gamma,
                     eps_start, eps_end, eps_decay,
                     progress_bar=None, status_text=None):
    Q          = np.zeros((env.n_states, env.n_actions))
    rewards    = []
    success    = []
    epsilon    = eps_start
    best_path  = None
    best_len   = 10_000
    report_at  = max(1, episodes // 40)

    for ep in range(episodes):
        state = env.reset()
        done  = False
        steps = 0
        ep_path = []

        while not done and steps < 300:
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = int(np.argmax(Q[state]))

            nstate, reward, done, tile, actual = env.step(action)
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[nstate]) - Q[state, action]
            )
            ep_path.append((state, action, actual, reward, done, tile))
            state = nstate
            steps += 1

        epsilon = max(eps_end, epsilon * eps_decay)
        won = ep_path and ep_path[-1][5] == "G"
        rewards.append(1.0 if won else 0.0)
        success.append(1 if won else 0)

        if won and len(ep_path) < best_len:
            best_path = ep_path[:]
            best_len  = len(ep_path)

        if progress_bar is not None and ep % max(1, episodes // 200) == 0:
            progress_bar.progress((ep + 1) / episodes)

        if status_text is not None and ep % report_at == 0:
            w   = min(500, ep + 1)
            wr  = np.mean(success[-w:]) * 100
            status_text.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Episode</div>'
                f'<div class="metric-value">{ep+1:,} / {episodes:,}</div>'
                f'<div class="metric-label" style="margin-top:8px;">Win rate (last {w})</div>'
                f'<div class="metric-value" style="font-size:1.4rem;">{wr:.1f}%</div>'
                f'<div class="metric-label" style="margin-top:6px;">ε = {epsilon:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    return Q, rewards, success, best_path


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────
def draw_grid(env, agent_state=None, path_states=None, Q=None):
    nrow, ncol = env.nrow, env.ncol
    fig, ax = plt.subplots(figsize=(max(4, ncol * 1.1 + 0.6),
                                    max(4, nrow * 1.1 + 0.6)))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    path_set = set(path_states) if path_states else set()

    for r in range(nrow):
        for c in range(ncol):
            tile  = env.desc[r][c]
            state = r * ncol + c
            dim   = path_states is not None and state not in path_set and state != agent_state
            al    = 0.35 if dim else 1.0

            ax.add_patch(mpatches.FancyBboxPatch(
                (c + 0.06, nrow - r - 0.94), 0.88, 0.88,
                boxstyle="round,pad=0.05",
                facecolor=TILE_COLOR[tile], edgecolor=TILE_BORDER[tile],
                linewidth=1.8, alpha=al, zorder=1,
            ))
            ax.text(c + 0.5, nrow - r - 0.5, TILE_EMOJI.get(tile, tile),
                    ha="center", va="center",
                    fontsize=max(10, 18 - ncol), zorder=3, alpha=al)

            if Q is not None and tile not in ("H", "G"):
                best_a = int(np.argmax(Q[state]))
                ddr, ddc = ACT_DELTA[best_a]
                ax.annotate("",
                    xy=(c + 0.5 + ddc * 0.30, nrow - r - 0.5 - ddr * 0.30),
                    xytext=(c + 0.5, nrow - r - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#00d4ff99", lw=1.4),
                    zorder=4, alpha=al)

            if path_states and state in path_set and state != agent_state:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c + 0.06, nrow - r - 0.94), 0.88, 0.88,
                    boxstyle="round,pad=0.05",
                    facecolor="none", edgecolor="#00ffcc",
                    linewidth=2.8, zorder=2, alpha=0.75,
                ))

    if agent_state is not None:
        ar, ac = env.pos_from_state(agent_state)
        ax.text(ac + 0.5, nrow - ar - 0.5, "🤖",
                ha="center", va="center",
                fontsize=max(12, 22 - ncol), zorder=5)

    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def draw_training_chart(rewards, success, window=100):
    w  = min(window, max(10, len(success) // 20))
    rs = np.convolve(success, np.ones(w) / w, mode="valid")
    rr = np.convolve(rewards, np.ones(w) / w, mode="valid")
    x  = np.arange(len(rs)) + w

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor("#0a0e1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#080c18")
        ax.tick_params(colors="#5a8ab0", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1a3550")

    ax1.plot(x, rs * 100, color="#00d4ff", lw=1.8)
    ax1.fill_between(x, rs * 100, alpha=0.13, color="#00d4ff")
    ax1.set_ylabel("Win Rate %", color="#8ab4d4", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", color="#1a3550", lw=0.5)
    ax1.set_title("Rolling Win Rate", color="#7dd4f8", fontsize=10, pad=4)

    ax2.plot(x, rr * 100, color="#00ffcc", lw=1.8)
    ax2.fill_between(x, rr * 100, alpha=0.13, color="#00ffcc")
    ax2.set_ylabel("Avg Win %", color="#8ab4d4", fontsize=9)
    ax2.set_xlabel("Episode", color="#8ab4d4", fontsize=9)
    ax2.grid(axis="y", color="#1a3550", lw=0.5)
    ax2.set_title("Rolling Reward Signal", color="#7dd4f8", fontsize=10, pad=4)

    fig.suptitle(f"Training Progress  (window = {w} eps)",
                 color="#7dd4f8", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def draw_qtable_heatmap(env, Q):
    nrow, ncol = env.nrow, env.ncol
    action_labels = ["← Left", "↓ Down", "→ Right", "↑ Up"]
    mask = np.zeros(Q.shape, dtype=bool)
    for r in range(nrow):
        for c in range(ncol):
            if env.desc[r][c] in ("H", "G"):
                mask[r * ncol + c, :] = True

    valid = Q[~mask[:, 0]]
    vmin  = valid.min() if len(valid) else -1
    vmax  = valid.max() if len(valid) else 1

    fig, axes = plt.subplots(1, 4, figsize=(max(12, ncol * 1.8), max(3, nrow * 0.9 + 1.5)))
    fig.patch.set_facecolor("#0a0e1a")

    for a_idx, ax in enumerate(axes):
        gq = Q[:, a_idx].reshape(nrow, ncol)
        mg = mask[:, a_idx].reshape(nrow, ncol)

        sns.heatmap(
            gq, ax=ax, mask=mg,
            cmap=sns.color_palette("mako_r", as_cmap=True),
            vmin=vmin, vmax=vmax,
            annot=True, fmt=".2f",
            annot_kws={"size": max(6, 9 - ncol // 2), "color": "white"},
            linewidths=0.4, linecolor="#1a3550",
            cbar=(a_idx == 3), square=True,
        )
        for r in range(nrow):
            for c in range(ncol):
                t = env.desc[r][c]
                if t in ("H", "G", "S"):
                    ax.text(c + 0.5, r + 0.5, TILE_EMOJI[t],
                            ha="center", va="center",
                            fontsize=max(7, 12 - ncol // 2))

        ax.set_title(action_labels[a_idx], color="#00d4ff", fontsize=10, pad=6)
        ax.set_facecolor("#080c18")
        ax.tick_params(colors="#5a8ab0", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1a3550")

    fig.suptitle("Q-Table Heatmap — Value per State × Action",
                 color="#7dd4f8", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
DEFAULTS = {
    "Q": None, "rewards": [], "success": [],
    "trained": False, "best_path": None,
    "map_snap": "4x4 Classic", "slip_snap": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❄️ Config")
    st.markdown("---")
    map_name    = st.selectbox("Grid Map", list(MAPS.keys()))
    is_slippery = st.toggle("Slippery Ice ❄️", value=True,
        help="Agent may slide sideways (1/3 chance each direction)")
    st.markdown("#### Q-Learning Hyperparams")
    episodes  = st.slider("Episodes",        1000, 30_000, 10_000, step=1000)
    alpha     = st.slider("Learning Rate α",  0.01,   1.0,    0.5,  step=0.01)
    gamma     = st.slider("Discount γ",       0.50, 0.999,   0.95,  step=0.001, format="%.3f")
    eps_start = st.slider("ε Start",           0.5,   1.0,    1.0,  step=0.05)
    eps_end   = st.slider("ε End",           0.001,  0.20,   0.05,  step=0.001, format="%.3f")
    eps_decay = st.slider("ε Decay",         0.990, 0.9999, 0.9995, step=0.0001, format="%.4f")
    st.markdown("---")
    replay_speed = st.slider("Replay Delay (s/step)", 0.1, 2.0, 0.5, step=0.1)
    st.markdown(
        '<p style="color:#2a5a8a;font-size:0.75rem;">'
        'Higher α → faster learning<br>Lower γ → prefer near rewards<br>'
        'Higher ε decay → slower exploration fade</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# ❄️ Frozen Lake RL Solver")
st.markdown(
    '<p style="color:#4a8ab0;font-size:0.92rem;margin-top:-14px;">'
    'Q-Learning · Real-time Training · Step-by-Step Replay · Reward Chart · Q-Table Heatmap'
    '</p>', unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.7], gap="large")

with col_left:
    st.markdown("### 🗺️ Grid World")
    env_preview = FrozenLakeEnv(map_name, is_slippery)
    grid_ph = st.empty()
    fig0 = draw_grid(env_preview)
    grid_ph.pyplot(fig0, use_container_width=True)
    plt.close(fig0)

    st.markdown(
        '<div style="font-size:0.9rem;margin-top:4px;">'
        '🟦 Start &nbsp; ⬜ Frozen &nbsp; 🕳️ Hole &nbsp; 🏆 Goal &nbsp; 🤖 Agent'
        '</div>', unsafe_allow_html=True,
    )
    st.markdown("")

    ma, mb, mc = st.columns(3)
    win_ph    = ma.empty()
    steps_ph  = mb.empty()
    result_ph = mc.empty()

    st.markdown("")
    train_btn = st.button("🚀  Train Agent", use_container_width=True)

with col_right:
    tab_train, tab_replay, tab_heatmap = st.tabs(
        ["📈 Training Progress", "🎬 Step-by-Step Replay", "🔥 Q-Table Heatmap"]
    )
    with tab_train:
        prog_ph   = st.progress(0)
        status_ph = st.empty()
        chart_ph  = st.empty()
    with tab_replay:
        rinfo_ph  = st.empty()
        rgrid_ph  = st.empty()
        rplay_btn = st.empty()
    with tab_heatmap:
        hmap_ph  = st.empty()
        hinfo_ph = st.empty()


# ─────────────────────────────────────────────
# Training logic
# ─────────────────────────────────────────────
if train_btn:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

    env = FrozenLakeEnv(map_name, is_slippery)
    st.session_state.map_snap  = map_name
    st.session_state.slip_snap = is_slippery

    with tab_train:
        status_ph.markdown(
            '<div class="metric-card"><div class="metric-label">Status</div>'
            '<div class="metric-value" style="font-size:1.2rem;">🔄 Training…</div></div>',
            unsafe_allow_html=True,
        )

    Q, rewards, success, best_path = train_q_learning(
        env, episodes, alpha, gamma,
        eps_start, eps_end, eps_decay,
        progress_bar=prog_ph, status_text=status_ph,
    )

    st.session_state.Q         = Q
    st.session_state.rewards   = rewards
    st.session_state.success   = success
    st.session_state.best_path = best_path
    st.session_state.trained   = True

    final_wr = np.mean(success[-min(1000, episodes):]) * 100
    won      = best_path is not None
    path_len = len(best_path) if won else 0

    win_ph.metric("Win Rate",    f"{final_wr:.1f}%")
    steps_ph.metric("Best Path", path_len if won else "—")
    result_ph.metric("Outcome",  "✅ Goal!" if won else "❌ No path")

    if best_path:
        ps  = {s[0] for s in best_path}
        fig_g = draw_grid(env, agent_state=best_path[-1][0], path_states=ps, Q=Q)
    else:
        fig_g = draw_grid(env, Q=Q)
    grid_ph.pyplot(fig_g, use_container_width=True)
    plt.close(fig_g)

    with tab_train:
        prog_ph.progress(1.0)
        status_ph.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Training Complete ✓</div>'
            f'<div class="metric-value">{episodes:,} episodes</div>'
            f'<div class="metric-label" style="margin-top:8px;">Win rate (last 1 000 eps)</div>'
            f'<div class="metric-value" style="font-size:1.5rem;color:#00ffcc;">{final_wr:.1f}%</div>'
            f'</div>', unsafe_allow_html=True,
        )
        fig_c = draw_training_chart(rewards, success)
        chart_ph.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    with tab_heatmap:
        fig_hm = draw_qtable_heatmap(env, Q)
        hmap_ph.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)
        hinfo_ph.markdown(
            '<p style="color:#4a7a9a;font-size:0.8rem;">'
            'Each cell = Q-value for that (state, action) pair. '
            'Grid arrows = greedy best action. Bright = high value; dark = low/negative.</p>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Replay (persists across Streamlit re-runs)
# ─────────────────────────────────────────────
if st.session_state.trained and st.session_state.Q is not None:
    env_r     = FrozenLakeEnv(st.session_state.map_snap,
                              is_slippery=st.session_state.slip_snap)
    best_path = st.session_state.best_path

    with tab_replay:
        if not best_path:
            rinfo_ph.warning(
                "Agent never reached the goal. "
                "Try more episodes, a higher α, or disable slippery ice."
            )
        else:
            won_tile = best_path[-1][5]
            ps_set   = {s[0] for s in best_path}

            rinfo_ph.markdown(
                f'<div class="metric-card" style="text-align:left;padding:10px 18px;">'
                f'<span style="color:#8ab4d4;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;">Best path found</span>'
                f' &nbsp;|&nbsp; <span style="font-family:Space Mono,monospace;color:#00d4ff;">{len(best_path)} steps</span>'
                f' &nbsp;|&nbsp; <span style="color:{"#00ff96" if won_tile=="G" else "#ff5050"};font-family:Space Mono,monospace;">'
                f'{"🏆 Goal!" if won_tile=="G" else "🕳️ Hole"}</span>'
                f'</div>', unsafe_allow_html=True,
            )

            fig_st = draw_grid(env_r, agent_state=best_path[-1][0],
                               path_states=ps_set, Q=st.session_state.Q)
            rgrid_ph.pyplot(fig_st, use_container_width=True)
            plt.close(fig_st)

            if rplay_btn.button("▶️  Play Step-by-Step Replay", use_container_width=True):
                for idx, (state, intended, actual, reward, done, tile) in enumerate(best_path):
                    fig_r = draw_grid(env_r, agent_state=state,
                                      path_states=ps_set, Q=st.session_state.Q)
                    rgrid_ph.pyplot(fig_r, use_container_width=True)
                    plt.close(fig_r)

                    rinfo_ph.markdown(
                        f'<div class="metric-card" style="text-align:left;padding:10px 18px;">'
                        f'<b style="color:#00d4ff;">Step {idx+1}/{len(best_path)}</b>'
                        f' · Intended: <b style="color:#7dd4f8;">{ACT_SYM[intended]}</b>'
                        f' · Actual: <b style="color:#00ffcc;">{ACT_SYM[actual]}</b>'
                        f' · Tile: {TILE_EMOJI.get(tile, tile)}'
                        f' · Reward: <b style="color:{"#00ff96" if reward>0 else "#ffaa00" if reward==0 else "#ff6060"}">{reward:+.2f}</b>'
                        f'</div>', unsafe_allow_html=True,
                    )
                    time.sleep(replay_speed)

                rinfo_ph.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:{"#00ff96" if won_tile=="G" else "#ff5050"};">'
                    f'{"🏆 Goal Reached!" if won_tile=="G" else "🕳️ Fell in a Hole"}'
                    f'</div></div>', unsafe_allow_html=True,
                )

    # Keep charts visible on re-run
    if st.session_state.rewards:
        with tab_train:
            fig_c2 = draw_training_chart(st.session_state.rewards, st.session_state.success)
            chart_ph.pyplot(fig_c2, use_container_width=True)
            plt.close(fig_c2)

    with tab_heatmap:
        fig_hm2 = draw_qtable_heatmap(env_r, st.session_state.Q)
        hmap_ph.pyplot(fig_hm2, use_container_width=True)
        plt.close(fig_hm2)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#1a3a5a;font-size:0.74rem;text-align:center;">'
    '❄️ Frozen Lake RL Solver · Q-Learning · Streamlit + NumPy + Matplotlib + Seaborn · No Gym required'
    '</p>', unsafe_allow_html=True,
)