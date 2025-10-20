# Terminal Graphing Calculator 🚀📈

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/downloads/)
[![SymPy](https://img.shields.io/badge/SymPy-1.12%2B-orange?logo=sympy)](https://www.sympy.org/en/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/yourusername/terminal-graphing-calc?style=social)](https://github.com/kaustubhxyz/Graphing-Calculator/)
[![Forks](https://img.shields.io/github/forks/yourusername/terminal-graphing-calc?style=social)](https://github.com/kaustubhxyz/Graphing-Calculator/)
[![Issues](https://img.shields.io/github/issues/yourusername/terminal-graphing-calc?style=flat)](https://github.com/kaustubhxyz/Graphing-Calculator/issues/)

A **powerful, fully interactive terminal-based graphing calculator**, built with Python, `curses` for the TUI, and `sympy` for symbolic math. Plot functions, parametric equations, polar curves, 3D projections, and more—all in your terminal! Supports live editing, animations, derivatives, intersections, and multi-color ASCII plots. No GUI needed—just pure text magic. 🌟

## ✨ Features

- **Multi-Mode Plotting**:
  - **Cartesian**: `y = f(x)` or `x = f(y)` (explicit/implicit).
  - **Polar**: `r = f(θ)` (e.g., roses, spirals).
  - **Parametric**: `x = f(t), y = g(t)` (e.g., Lissajous curves).
  - **3D Projections**: `z = f(x,y)` (orthographic view with shading).

- **Advanced Math Tools**:
  - **Derivatives & Integrals**: Compute and display `f'(x)` symbolically.
  - **Intersections**: Find & mark points between curves (up to 5).
  - **Equation Solving**: Built-in `solve` for roots/intersections.
  - **Symbolic Evaluation**: Handles `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, and more.

- **Interactive UI**:
  - Live editing of expressions with cursor navigation.
  - Parameter substitution (e.g., `a=1.5`, `phase=0.0`) for dynamic plots.
  - Values table at key points (customizable for trig: 0, π/2, etc.).
  - Multi-color ASCII plots (up to 3 expressions with markers: `*`, `o`, `+`).
  - Zoom/pan with `+/-` and `L/R` keys.

- **Animations & Examples**:
  - **Wave Animations**: Toggle with `W`—phase shifts, amplitude/freq modulation (e.g., breathing Gaussians, expanding circles).
  - **Example Library**: 30+ pre-loaded examples across categories: waves, polar, parametric, implicit, 3D, functions. Cycle with `E`/`C`.
    - Waves: `sin(x + phase) + 0.5*sin(2*x - phase)`.
    - Polar: `r=1 + sin(5*t + phase)` (animated rose).
    - Implicit: `(x**2 + y**2)**2 = a*(x**2 - y**2)` (lemniscate).
    - 3D: `z=sin(sqrt(x**2 + y**2) + phase)` (ripples).

- **Performance Optimizations**:
  - Lambdified evaluation for speed (up to 10x faster than raw `subs`).
  - Adaptive sampling for implicit/3D (reduces from 200+ to 40-80 points based on zoom).
  - Bounds filtering to avoid infinities/NaNs.

- **Cross-Platform**: Works on Linux/Mac (native `curses`), Windows (via `pip install windows-curses`).

## 🛠️ Installation

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/kaustubhxyz/Graphing-Calculator.git
   cd Graphing-Calculator
   ```

2. **Install Dependencies** (Python 3.8+ required):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run**:
   ```bash
   python3 graphing_calculator.py
   ```
   - Ensure your terminal supports UTF-8 and colors (e.g., iTerm2, GNOME Terminal, or Windows Terminal).

## 🎮 Usage & Controls

Launch the app and start plotting! Here's a quick guide:

### Navigation & Editing
- **Arrows**: Move cursor/select lines (edit mode) or pan plot (view mode).
- **Space**: Toggle edit mode (✎).
- **Type/Backspace/Del**: Edit expressions live—updates plot instantly!
- **Enter**: Save edits or add new line (max 10).
- **X**: Delete current line.

### Plot Controls
- **+/-**: Zoom in/out (0.4x/1.25x range).
- **L/R**: Pan left/right (20% shift).
- **0**: Reset view to defaults (-2π to 2π, -2.5 to 2.5).

### Special Features
- **M**: Cycle modes (Cartesian → Polar → Parametric → 3D).
- **W**: Toggle animation (phase/amplitude waves—interrupt with W/ESC).
- **E**: Load next example in category.
- **C**: Switch categories (waves → polar → parametric → implicit → 3D → functions).
- **T**: Toggle values table.
- **D**: Toggle derivative display.
- **I**: Toggle intersections (marks ● on plot).
- **P**: Toggle params view.
- **ESC**: Cancel edit or quit.

### Example Workflow
1. Press `C` → "waves" → `E` (loads `sin(x + phase) + 0.5*sin(2*x - phase)`).
2. Press `W`—watch the superposition animate! 🎬
3. Edit to `a*sin(b*x + phase)` → `Space` to save → `P` → type `a=2` in new line + Enter.
4. `M` → "polar" → `E` (rose curve) → `W` for spinning petals.

**Pro Tip**: For implicit curves, try `(x**2 + y**2 - phase)**2 = a*(x**2 - y**2)`—`W` creates a hypnotic lemniscate animation!

## 📱 Screenshots

| Cartesian Waves (Animated) | Polar Rose Curve | Parametric Lissajous |
|--------------------|--------------------|-------------------|
| ![Waves](screenshots/waves.gif) | ![Polar](screenshots/polar_rose.gif) | ![Lissajous](screenshots/lissajous.gif) |

| Implicit Lemniscate | 3D Ripple Projection |
|--------------------|----------------------|
| ![Implicit](screenshots/lemniscate.gif) | ![3D](screenshots/3d_ripple.gif) |

*(GIFs captured via `asciinema` or screen record—add your own!)*

## 🔍 Troubleshooting

- **No Colors?** Run in a terminal with 256-color support (e.g., `export TERM=xterm-256color`).
- **SymPy Errors?** Update: `pip install --upgrade sympy`. Ensure expressions use valid syntax (e.g., `**` for power, `pi` for π).
- **Slow Animations?** Reduce `anim_speed = 0.03` in code or lower terminal resolution.
- **Windows Issues?** Install `windows-curses` and run in Windows Terminal.

## 🤝 Contributing

Love terminal math? Contribute!
1. Fork the repo & create a feature branch (`git checkout -b feat/amazing-wave`).
2. Commit changes (`git commit -m "Add: Fourier series support"`).
3. Push & open a PR!

Ideas: More examples, inequality shading, export to SVG, or ML curve fitting. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **SymPy**: For symbolic powerhouse math. 🧮
- **Curses**: Timeless TUI library.
- Inspired by Desmos, ASCII art plots, and terminal hackers everywhere.

**Star this repo if it sparks joy!** ⭐ Questions? Open an issue. Happy plotting! 🎉

---

*Built with ❤️ for math nerds in terminals. Last updated: October 18, 2025.*
