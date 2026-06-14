# 📈 LR Scheduler Playground

An interactive, browser-based explorer for learning rate schedulers — visualize, compare, and simulate how different schedules affect training dynamics.

![Demo](https://img.shields.io/badge/demo-live-brightgreen) ![No build](https://img.shields.io/badge/build-none-blue) ![Vanilla JS](https://img.shields.io/badge/stack-HTML%20%2F%20CSS%20%2F%20JS-orange)

---

▶ Live: [balayogig/learning-rate-scheduler](https://balayogig.github.io/learning-rate-scheduler/) — no install, nothing to set up.

Co-developed by Balayogi G & Claude.

## Features

### 📊 Visualizer
- Interactive LR curve for any scheduler with live parameter controls
- Rendered **math formula** (via MathJax) and explanations for each scheduler
- **Training phase bar** showing explore vs. converge balance
- Export the chart as **PNG**, **JSON**, or **CSV**
- Scroll / pinch to zoom, drag to pan, Reset Zoom button

### ⚖️ Compare
- Compare up to **4 schedulers** side-by-side on the same chart
- Per-scheduler stats (peak LR, final LR, area under curve)
- Zoom / pan and Reset Zoom

### 🏋️ Training Sim
- Animated **training & validation loss** simulation driven by the active schedule
- **Live Learning Rate** chart updates in sync
- Start / Pause / Reset controls
- Reset Zoom on both charts

### 🎯 Optimization
- **2D loss landscape** canvas (`f(x,y) = x² + 2y²`) with color heatmap and contour lines
- Optimizer ball rolls down the surface in real time, leaving a trail
- **Loss over epochs** line chart with zoom / pan and Reset Zoom
- Start / Pause / Reset controls

---

## Schedulers

| Scheduler | Key Parameters |
|---|---|
| Constant | Initial LR |
| Step Decay | Initial LR, Step Size, Gamma |
| Exponential Decay | Initial LR, Gamma |
| Polynomial Decay | Initial LR, Min LR, Power |
| Cosine Annealing | Initial LR, Min LR |
| Cosine Annealing w/ Warm Restarts | Initial LR, Min LR, Restart Period |
| Cyclic LR (Triangular) | Min LR, Max LR, Cycle Length |
| OneCycle | Initial LR, Max LR, Min LR, Cycle Length |
| Warmup + Cosine | Initial LR, Min LR, Warmup Epochs |

### Presets

| Preset | Scheduler | Description |
|---|---|---|
| ResNet | Step Decay | Standard ImageNet schedule (γ=0.1 every 30 epochs) |
| BERT | Warmup + Cosine | Transformer fine-tuning with 5-epoch warmup |
| GAN | Cyclic LR | Short cycles for adversarial stability |
| OneCycle Fast | OneCycle | Super-convergence in 60 epochs |

---

## Getting Started

No build step required — open `index.html` directly in any modern browser.

```bash
git clone https://github.com/your-username/learning-rate-scheduler.git
cd learning-rate-scheduler
open index.html        # macOS
xdg-open index.html    # Linux
start index.html       # Windows
```

Or serve locally:

```bash
npx serve .
# or
python3 -m http.server 8000
```

---

## Project Structure

```
learning-rate-scheduler/
├── index.html      # App shell, tab layout, all HTML
├── script.js       # All scheduler logic, chart setup, simulation
├── style.css       # Theming, layout, dark/light mode
└── favicon.svg     # Browser tab icon
```

---

## Dependencies (CDN, no install)

| Library | Version | Purpose |
|---|---|---|
| [Chart.js](https://www.chartjs.org/) | 4.4.0 | All line / scatter charts |
| [chartjs-plugin-zoom](https://www.chartjs.org/chartjs-plugin-zoom/) | 2.0.1 | Scroll zoom and pan on charts |
| [Hammer.js](https://hammerjs.github.io/) | 2.0.8 | Touch / pinch gesture support |
| [MathJax](https://www.mathjax.org/) | 3 | LaTeX formula rendering |

---

## Usage Tips

- **Sidebar sliders** update the chart in real time — no submit button needed
- **Scroll** on any chart to zoom in; **drag** to pan; click **Reset Zoom** to restore
- **Compare tab**: use the four dropdowns to pick any combination of schedulers
- **Training Sim**: watch how a smooth schedule (Cosine) vs. a step schedule affects val loss
- **Optimization**: run the same scheduler in Optimization to see how it navigates the loss bowl
- **Theme toggle** (🌙 / ☀️) in the top-right switches between dark and light mode
- **Collapse the sidebar** with the ‹ button to give charts more space
