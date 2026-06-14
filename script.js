/* ============================================================
   Learning Rate Scheduler Playground — script.js
   ============================================================ */
(function () {
  'use strict';

  // ==========================================================
  // SCHEDULER PURE FUNCTIONS  getLR(epoch, params) -> lr
  // ==========================================================
  const SCHED = {
    constant:      (e, p) => p.initialLR,
    step:          (e, p) => p.initialLR * Math.pow(p.gamma, Math.floor(e / p.stepSize)),
    exponential:   (e, p) => p.initialLR * Math.pow(p.gamma, e),
    polynomial:    (e, p) => {
      const pct = Math.max(0, 1 - e / Math.max(p.epochs - 1, 1));
      return (p.initialLR - p.minLR) * Math.pow(pct, p.power) + p.minLR;
    },
    cosine:        (e, p) => p.minLR + 0.5 * (p.initialLR - p.minLR) * (1 + Math.cos(Math.PI * e / p.epochs)),
    cosine_warm:   (e, p) => {
      const T = e % p.restartPeriod;
      return p.minLR + 0.5 * (p.initialLR - p.minLR) * (1 + Math.cos(Math.PI * T / p.restartPeriod));
    },
    cyclic:        (e, p) => {
      const cycle = Math.floor(1 + e / (2 * p.cycleLength));
      const x = Math.abs(e / p.cycleLength - 2 * cycle + 1);
      return p.minLR + (p.maxLR - p.minLR) * Math.max(0, 1 - x);
    },
    onecycle:      (e, p) => {
      const cl = p.cycleLength, half = cl / 2;
      if (e <= half) return p.initialLR + (p.maxLR - p.initialLR) * (e / half);
      if (e <= cl)   return p.minLR + (p.maxLR - p.minLR) * (1 - (e - half) / half);
      const rem = Math.max(p.epochs - cl, 1);
      return p.minLR * (1 - 0.9 * (e - cl) / rem);
    },
    warmup_cosine: (e, p) => {
      if (e < p.warmupEpochs) return p.initialLR * (e + 1) / (p.warmupEpochs + 1);
      const t = (e - p.warmupEpochs) / Math.max(p.epochs - p.warmupEpochs, 1);
      return p.minLR + 0.5 * (p.initialLR - p.minLR) * (1 + Math.cos(Math.PI * t));
    }
  };

  function getLR(name, epoch, params) {
    const fn = SCHED[name];
    if (!fn) return params.initialLR;
    return Math.max(0, fn(Math.max(0, epoch), params));
  }

  function genData(name, params) {
    const data = [];
    for (let e = 0; e <= params.epochs; e++) data.push({ x: e, y: getLR(name, e, params) });
    return data;
  }

  // ==========================================================
  // SCHEDULER METADATA
  // ==========================================================
  const META = {
    constant: {
      label: 'Constant',
      formula: `<div class="formula-main">$$\\alpha_t = \\alpha_0$$</div>`,
      vars: ['\\(\\alpha_0\\): initial learning rate'],
      desc: 'Keeps the learning rate fixed throughout all training epochs.',
      useCases: 'Baselines, fine-tuning with a pre-tuned LR, short experiments',
      pros: ['Simple', 'Predictable', 'No hyperparameters to tune'],
      cons: ['May oscillate near minimum', 'Hard to set the right value'],
      recommend: 'Use when you already know the optimal LR or for quick baselines.',
      params: ['initialLR', 'epochs']
    },
    step: {
      label: 'Step Decay',
      formula: `<div class="formula-main">$$\\alpha_t = \\alpha_0 \\cdot \\gamma^{\\lfloor t / s \\rfloor}$$</div>`,
      vars: ['\\(\\alpha_0\\): initial LR', '\\(\\gamma\\): decay factor (0–1)', '\\(s\\): step size (epochs)'],
      desc: 'Reduces the LR by a factor γ every s epochs. Simple and widely used.',
      useCases: 'ResNets, VGGs, image classification with known training schedules',
      pros: ['Intuitive', 'Easy to tune', 'Works well for image tasks'],
      cons: ['Abrupt LR drops can destabilize training', 'Requires manual tuning of step size'],
      recommend: 'Use for image classification when you have a large dataset and training for many epochs.',
      params: ['initialLR', 'epochs', 'stepSize', 'gamma']
    },
    exponential: {
      label: 'Exponential Decay',
      formula: `<div class="formula-main">$$\\alpha_t = \\alpha_0 \\cdot \\gamma^t$$</div>`,
      vars: ['\\(\\alpha_0\\): initial LR', '\\(\\gamma\\): decay factor per epoch (e.g. 0.95)'],
      desc: 'Continuously decays the LR exponentially. Smoother than step decay.',
      useCases: 'NLP tasks, RNNs, tasks needing smooth LR reduction',
      pros: ['Smooth decay', 'Simple formula'],
      cons: ['LR can become too small too quickly with small γ', 'No floor without clipping'],
      recommend: 'Use γ ≈ 0.95–0.99 for gradual decay. Pair with a minLR clip.',
      params: ['initialLR', 'epochs', 'gamma']
    },
    polynomial: {
      label: 'Polynomial Decay',
      formula: `<div class="formula-main">$$\\alpha_t = (\\alpha_0 - \\alpha_{min}) \\cdot \\left(1 - \\frac{t}{T}\\right)^p + \\alpha_{min}$$</div>`,
      vars: ['\\(\\alpha_0\\): initial LR', '\\(\\alpha_{min}\\): minimum LR', '\\(T\\): total epochs', '\\(p\\): polynomial power'],
      desc: 'Decays LR as a polynomial from initial to minimum. Power controls the decay shape.',
      useCases: 'BERT, transformer fine-tuning, NLP tasks',
      pros: ['Flexible curve shape via power', 'Guaranteed minimum LR'],
      cons: ['More hyperparameters than step decay'],
      recommend: 'Power=1 for linear decay (BERT-style), power=2 for quadratic.',
      params: ['initialLR', 'minLR', 'epochs', 'power']
    },
    cosine: {
      label: 'Cosine Annealing',
      formula: `<div class="formula-main">$$\\alpha_t = \\alpha_{min} + \\frac{1}{2}(\\alpha_0 - \\alpha_{min})\\left(1 + \\cos\\frac{\\pi t}{T}\\right)$$</div>`,
      vars: ['\\(\\alpha_0\\): initial LR', '\\(\\alpha_{min}\\): minimum LR', '\\(T\\): total epochs'],
      desc: 'Anneals LR following a cosine curve—fast initial drop, slow near minimum.',
      useCases: 'Deep learning in general, vision transformers, competitive training runs',
      pros: ['Smooth transitions', 'Naturally spends more time at extreme LRs', 'Strong empirical performance'],
      cons: ['Single cycle—restarts needed for longer training'],
      recommend: 'Default choice for most tasks. Pair with warmup for transformers.',
      params: ['initialLR', 'minLR', 'epochs']
    },
    cosine_warm: {
      label: 'Cosine w/ Warm Restarts',
      formula: `<div class="formula-main">$$\\alpha_t = \\alpha_{min} + \\frac{1}{2}(\\alpha_0 - \\alpha_{min})\\left(1 + \\cos\\frac{\\pi (t \\bmod T_r)}{T_r}\\right)$$</div>`,
      vars: ['\\(T_r\\): restart period', '\\(t \\bmod T_r\\): epoch within current cycle'],
      desc: 'Cosine annealing with periodic restarts. Each restart escapes local minima.',
      useCases: 'Ensemble building, long training runs, exploring loss landscape',
      pros: ['Escapes local minima', 'Can snapshot ensemble at each restart'],
      cons: ['Restarts can be disruptive', 'More hyperparameters'],
      recommend: 'Use for long runs. Set T_r to 10–30% of total epochs.',
      params: ['initialLR', 'minLR', 'epochs', 'restartPeriod']
    },
    cyclic: {
      label: 'Cyclic LR (Triangular)',
      formula: `<div class="formula-main">$$x = \\left|\\frac{t}{C} - 2 \\cdot \\text{cycle} + 1\\right|, \\quad \\alpha_t = \\alpha_{min} + (\\alpha_{max}-\\alpha_{min})\\cdot\\max(0,1-x)$$</div>`,
      vars: ['\\(C\\): cycle length', '\\(\\text{cycle} = \\lfloor 1 + t/(2C) \\rfloor\\)'],
      desc: 'Cycles LR between min and max in triangular waves. Helps escape saddle points.',
      useCases: 'GANs, tasks with rugged loss landscapes, fast training experiments',
      pros: ['Escapes local minima', 'Eliminates LR sensitivity', 'Can train faster'],
      cons: ['Oscillating LR can slow final convergence'],
      recommend: 'Set cycle length to 2–10% of total epochs. Good for GANs.',
      params: ['minLR', 'maxLR', 'epochs', 'cycleLength']
    },
    onecycle: {
      label: 'OneCycle',
      formula: `<div class="formula-main">$$\\text{Phase 1: }\\alpha_0 \\to \\alpha_{max} \\;\\text{ (half cycle)}$$$$\\text{Phase 2: }\\alpha_{min} \\to \\alpha_{max} \\;\\text{ (half cycle)}$$</div>`,
      vars: ['\\(\\alpha_0\\): start LR', '\\(\\alpha_{max}\\): peak LR', '\\(\\alpha_{min}\\): final LR'],
      desc: 'Ramps LR up then down within one cycle, then anneals to near-zero.',
      useCases: 'Fast training (super-convergence), image classification, ResNets',
      pros: ['Super-convergence possible', 'Strong regularization effect', 'Often needs fewer epochs'],
      cons: ['Requires careful max_lr selection'],
      recommend: 'Use lr_finder to find max_lr. Set cycle to 80–90% of total epochs.',
      params: ['initialLR', 'maxLR', 'minLR', 'epochs', 'cycleLength']
    },
    warmup_cosine: {
      label: 'Warmup + Cosine',
      formula: `<div class="formula-main">$$\\alpha_t = \\begin{cases} \\alpha_0 \\cdot \\dfrac{t+1}{W+1} & t < W \\\\ \\alpha_{min} + \\dfrac{\\alpha_0-\\alpha_{min}}{2}\\left(1+\\cos\\dfrac{\\pi(t-W)}{T-W}\\right) & t \\geq W \\end{cases}$$</div>`,
      vars: ['\\(W\\): warmup epochs', '\\(T\\): total epochs'],
      desc: 'Linearly warms up LR then applies cosine annealing. Standard for transformers.',
      useCases: 'Transformers, BERT, GPT, ViT, any large-scale model training',
      pros: ['Stable early training via warmup', 'Smooth cosine convergence'],
      cons: ['Warmup period adds training time before peak performance'],
      recommend: 'Standard choice for all transformer models. W ≈ 1–5% of T.',
      params: ['initialLR', 'minLR', 'epochs', 'warmupEpochs']
    }
  };

  // ==========================================================
  // PARAMETER DEFINITIONS
  // ==========================================================
  const PARAM_DEFS = {
    initialLR:     { label: 'Initial LR (α₀)', min: 0.0001, max: 1.0,  step: 0.0001, def: 0.1,   fmt: '4f', tip: 'Starting learning rate' },
    minLR:         { label: 'Min LR (α_min)',   min: 0.00001,max: 0.1,  step: 0.00001,def: 0.001, fmt: '5f', tip: 'Minimum learning rate floor' },
    maxLR:         { label: 'Max LR (α_max)',   min: 0.001,  max: 1.0,  step: 0.001,  def: 0.1,   fmt: '4f', tip: 'Maximum learning rate ceiling' },
    epochs:        { label: 'Epochs (T)',        min: 10,     max: 500,  step: 1,      def: 100,   fmt: 'int',tip: 'Total training epochs' },
    stepSize:      { label: 'Step Size (s)',     min: 1,      max: 100,  step: 1,      def: 20,    fmt: 'int',tip: 'Epochs between LR drops' },
    gamma:         { label: 'Gamma (γ)',         min: 0.01,   max: 0.99, step: 0.01,   def: 0.5,   fmt: '2f', tip: 'Decay factor per step' },
    warmupEpochs:  { label: 'Warmup Epochs (W)',min: 1,      max: 100,  step: 1,      def: 10,    fmt: 'int',tip: 'Epochs for linear LR warmup' },
    restartPeriod: { label: 'Restart Period (Tᵣ)',min:5,     max: 200,  step: 1,      def: 30,    fmt: 'int',tip: 'Epochs between cosine restarts' },
    cycleLength:   { label: 'Cycle Length (C)', min: 5,      max: 200,  step: 1,      def: 30,    fmt: 'int',tip: 'Epochs per LR cycle' },
    power:         { label: 'Power (p)',         min: 0.1,    max: 5.0,  step: 0.1,    def: 1.0,   fmt: '1f', tip: 'Polynomial decay power' }
  };

  function fmtVal(val, fmt) {
    if (fmt === 'int') return String(Math.round(val));
    const d = parseInt(fmt) || 4;
    return val.toFixed(d);
  }

  // ==========================================================
  // PRESETS
  // ==========================================================
  const PRESETS = {
    resnet:  { sched: 'step',         params: { initialLR:0.1,   minLR:0.0001, epochs:200, stepSize:30, gamma:0.1  } },
    bert:    { sched: 'warmup_cosine',params: { initialLR:0.0005,minLR:0.00001,epochs:50,  warmupEpochs:5 } },
    gan:     { sched: 'cyclic',       params: { minLR:0.0002,    maxLR:0.002,  epochs:100, cycleLength:10 } },
    fast:    { sched: 'onecycle',     params: { initialLR:0.01,  maxLR:0.1,    minLR:0.001,epochs:60,   cycleLength:50 } }
  };

  // ==========================================================
  // APPLICATION STATE
  // ==========================================================
  const S = {
    scheduler: 'cosine',
    params: Object.fromEntries(Object.entries(PARAM_DEFS).map(([k, v]) => [k, v.def])),
    theme: 'dark',
    activeTab: 'visualizer',
    comparison: ['cosine', 'step', 'warmup_cosine', 'cyclic'],
    training: { running: false, epoch: 0, timer: null, trainData: [], valData: [], lrData: [] },
    opt: { running: false, epoch: 0, timer: null, x: 2.5, y: 1.8, trail: [], lossData: [] },
  };

  // ==========================================================
  // CHART INSTANCES
  // ==========================================================
  const CHARTS = {};
  const COLORS = ['#4f9cf9', '#f97316', '#22c55e', '#a855f7'];

  function getThemeColors() {
    const dark = S.theme === 'dark';
    return {
      grid:  dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)',
      tick:  dark ? '#94a3b8' : '#64748b',
      bg:    dark ? '#0f172a' : '#f1f5f9'
    };
  }

  function baseChartCfg(extra = {}) {
    const tc = getThemeColors();
    return {
      type: 'line',
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300, easing: 'easeInOutQuad' },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: tc.tick, boxWidth: 14, font: { size: 11 } } },
          tooltip: { callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toExponential(3)}`
          }},
          zoom: {
            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
            pan:  { enabled: true, mode: 'xy' }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Epoch', color: tc.tick, font: { size: 11 } },
            grid: { color: tc.grid },
            ticks: { color: tc.tick, font: { size: 10 } }
          },
          y: {
            title: { display: true, text: 'Learning Rate', color: tc.tick, font: { size: 11 } },
            grid: { color: tc.grid },
            ticks: { color: tc.tick, font: { size: 10 }, callback: v => v.toExponential(1) }
          }
        },
        ...extra
      }
    };
  }

  function initCharts() {
    const tc = getThemeColors();

    // Main LR chart
    const lrCtx = document.getElementById('lrChart').getContext('2d');
    CHARTS.lr = new Chart(lrCtx, {
      ...baseChartCfg(),
      data: { datasets: [{ label: 'Learning Rate', data: [], borderColor: COLORS[0], backgroundColor: hexAlpha(COLORS[0], 0.12), borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 }] }
    });

    // Comparison chart
    const cmpCtx = document.getElementById('comparisonChart').getContext('2d');
    CHARTS.cmp = new Chart(cmpCtx, {
      ...baseChartCfg({ plugins: { zoom: { zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }, pan: { enabled: true, mode: 'xy' } } } }),
      data: {
        datasets: COLORS.map((c, i) => ({
          label: META[S.comparison[i]]?.label || '—',
          data: [], borderColor: c, backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3
        }))
      }
    });

    // Training charts
    const trCtx = document.getElementById('trainingChart').getContext('2d');
    CHARTS.train = new Chart(trCtx, {
      ...baseChartCfg(),
      data: {
        datasets: [
          { label: 'Train Loss', data: [], borderColor: COLORS[0], backgroundColor: hexAlpha(COLORS[0], 0.1), borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 },
          { label: 'Val Loss',   data: [], borderColor: COLORS[1], backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [5,3] }
        ]
      },
      options: { ...baseChartCfg().options, scales: { ...baseChartCfg().options.scales, y: { ...baseChartCfg().options.scales.y, title: { display: true, text: 'Loss', color: tc.tick }, ticks: { color: tc.tick, font: { size: 10 } } } } }
    });

    const llCtx = document.getElementById('liveLRChart').getContext('2d');
    CHARTS.liveLR = new Chart(llCtx, {
      ...baseChartCfg(),
      data: { datasets: [{ label: 'LR', data: [], borderColor: COLORS[2], backgroundColor: hexAlpha(COLORS[2], 0.15), borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 }] }
    });

    // Opt loss chart
    const olCtx = document.getElementById('optLossChart').getContext('2d');
    CHARTS.optLoss = new Chart(olCtx, {
      ...baseChartCfg(),
      data: { datasets: [{ label: 'Loss', data: [], borderColor: COLORS[3], backgroundColor: hexAlpha(COLORS[3], 0.15), borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 }] },
      options: { ...baseChartCfg().options, scales: { ...baseChartCfg().options.scales, y: { ...baseChartCfg().options.scales.y, title: { display: true, text: 'Loss f(x,y)', color: tc.tick } } } }
    });

  }

  function updateMainChart() {
    const data = genData(S.scheduler, S.params);
    CHARTS.lr.data.datasets[0].data = data;
    CHARTS.lr.data.datasets[0].label = META[S.scheduler]?.label || S.scheduler;
    CHARTS.lr.update('active');
    updatePhaseBar(data);
  }

  function updateComparisonChart() {
    S.comparison.forEach((sched, i) => {
      const meta = META[sched];
      const data = genData(sched, S.params);
      CHARTS.cmp.data.datasets[i].data = data;
      CHARTS.cmp.data.datasets[i].label = meta?.label || sched;
    });
    CHARTS.cmp.update('active');
    renderComparisonStats();
  }

  // ==========================================================
  // PARAMETER CONTROLS
  // ==========================================================
  function renderControls() {
    const container = document.getElementById('paramControls');
    const keys = META[S.scheduler]?.params || Object.keys(PARAM_DEFS);
    container.innerHTML = keys.map(k => {
      const d = PARAM_DEFS[k];
      if (!d) return '';
      const v = S.params[k];
      return `<div class="param-row">
        <label>
          <span title="${d.tip}">${d.label}</span>
          <span class="param-val" id="val_${k}">${fmtVal(v, d.fmt)}</span>
        </label>
        <input type="range" id="range_${k}" min="${d.min}" max="${d.max}" step="${d.step}" value="${v}">
        <div class="param-info">${d.tip}</div>
      </div>`;
    }).join('');

    keys.forEach(k => {
      const el = document.getElementById(`range_${k}`);
      if (!el) return;
      el.addEventListener('input', () => {
        S.params[k] = parseFloat(el.value);
        document.getElementById(`val_${k}`).textContent = fmtVal(S.params[k], PARAM_DEFS[k].fmt);
        onParamChange();
      });
    });
  }

  function onParamChange() {
    updateMainChart();
    if (S.activeTab === 'comparison') updateComparisonChart();
    renderFormula();
  }

  // ==========================================================
  // FORMULA PANEL
  // ==========================================================
  function renderFormula() {
    const m = META[S.scheduler];
    if (!m) return;
    const el = document.getElementById('formulaContent');
    el.innerHTML = m.formula + `<ul class="formula-vars">${m.vars.map(v => `<li>${v}</li>`).join('')}</ul>`;
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([el]).catch(() => {});
    }
  }

  function renderExplanation() {
    const m = META[S.scheduler];
    if (!m) return;
    document.getElementById('explanationContent').innerHTML = `
      <div class="exp-section"><h3>Description</h3><p>${m.desc}</p></div>
      <div class="exp-section"><h3>Use Cases</h3><p>${m.useCases}</p></div>
      <div class="exp-section"><h3>Advantages</h3><div class="exp-tags">${m.pros.map(p => `<span class="exp-tag pro">✓ ${p}</span>`).join('')}</div></div>
      <div class="exp-section"><h3>Disadvantages</h3><div class="exp-tags">${m.cons.map(c => `<span class="exp-tag con">✗ ${c}</span>`).join('')}</div></div>
      <div class="exp-section"><h3>Recommendation</h3><p>${m.recommend}</p></div>`;
  }

  // ==========================================================
  // PHASE BAR
  // ==========================================================
  function updatePhaseBar(data) {
    if (!data || !data.length) return;
    const lrs = data.map(d => d.y);
    const maxLR = Math.max(...lrs), minLR = Math.min(...lrs);
    const range = maxLR - minLR || 1;
    const highThresh = minLR + range * 0.6, lowThresh = minLR + range * 0.25;
    let explore = 0, converge = 0, mid = 0;
    lrs.forEach(lr => {
      if (lr >= highThresh) explore++;
      else if (lr <= lowThresh) converge++;
      else mid++;
    });
    const total = lrs.length;
    const ep = (explore / total * 100).toFixed(0);
    const mp = (mid / total * 100).toFixed(0);
    const cp = (converge / total * 100).toFixed(0);
    document.getElementById('phaseBarInner').innerHTML = `
      <div style="width:${ep}%;background:linear-gradient(90deg,#f97316,#f59e0b)"></div>
      <div style="width:${mp}%;background:#334155"></div>
      <div style="width:${cp}%;background:linear-gradient(90deg,#3b82f6,#1d4ed8)"></div>`;
  }

  // ==========================================================
  // COMPARISON MODE
  // ==========================================================
  function initComparisonUI() {
    const container = document.getElementById('comparisonSelectors');
    const schedKeys = Object.keys(META);
    container.innerHTML = S.comparison.map((sel, i) => `
      <div class="comp-slot">
        <div class="comp-slot-label">
          <span class="comp-color-dot" style="background:${COLORS[i]}"></span>
          Slot ${i + 1}
        </div>
        <select class="select-input" id="comp-sel-${i}">
          ${schedKeys.map(k => `<option value="${k}"${k === sel ? ' selected' : ''}>${META[k].label}</option>`).join('')}
        </select>
      </div>`).join('');

    S.comparison.forEach((_, i) => {
      document.getElementById(`comp-sel-${i}`).addEventListener('change', e => {
        S.comparison[i] = e.target.value;
        updateComparisonChart();
      });
    });
    updateComparisonChart();
  }

  function renderComparisonStats() {
    const container = document.getElementById('comparisonStats');
    container.innerHTML = S.comparison.map((sched, i) => {
      const data = genData(sched, S.params).map(d => d.y);
      const finalLR = data[data.length - 1];
      const avgLR   = data.reduce((a, b) => a + b, 0) / data.length;
      const maxLRv  = Math.max(...data);
      const minLRv  = Math.min(...data);
      return `<div class="stat-card" style="border-top: 2px solid ${COLORS[i]}">
        <h4>${META[sched]?.label}</h4>
        <div class="stat-row"><span>Final LR</span><span>${finalLR.toExponential(2)}</span></div>
        <div class="stat-row"><span>Average LR</span><span>${avgLR.toExponential(2)}</span></div>
        <div class="stat-row"><span>Max LR</span><span>${maxLRv.toExponential(2)}</span></div>
        <div class="stat-row"><span>Min LR</span><span>${minLRv.toExponential(2)}</span></div>
      </div>`;
    }).join('');
  }

  // ==========================================================
  // TRAINING SIMULATOR
  // ==========================================================
  function trainingReset() {
    if (S.training.timer) { clearInterval(S.training.timer); S.training.timer = null; }
    S.training.epoch = 0; S.training.running = false;
    S.training.trainData = []; S.training.valData = []; S.training.lrData = [];
    CHARTS.train.data.datasets[0].data = [];
    CHARTS.train.data.datasets[1].data = [];
    CHARTS.liveLR.data.datasets[0].data = [];
    CHARTS.train.update(); CHARTS.liveLR.update();
    document.getElementById('trainingStats').innerHTML = '';
    document.getElementById('startTraining').disabled = false;
    document.getElementById('pauseTraining').disabled = true;
  }

  function trainingStep() {
    const e = S.training.epoch;
    const ep = S.params.epochs;
    if (e > ep) { trainingPause(); return; }

    const lr    = getLR(S.scheduler, e, S.params);
    const initLR = S.params.initialLR || 0.1;
    const t      = e / Math.max(ep, 1);
    const baseL  = 0.8 * Math.exp(-3.5 * t) + 0.15;
    const noise  = () => (Math.random() - 0.5) * 0.04;
    const lrFrac = lr / initLR;
    const trainL = Math.max(0.05, baseL * (1 + noise()));
    const valL   = Math.max(0.06, trainL + 0.05 * lrFrac + noise() * 1.5 + 0.02);

    S.training.trainData.push({ x: e, y: trainL });
    S.training.valData.push({ x: e, y: valL });
    S.training.lrData.push({ x: e, y: lr });

    CHARTS.train.data.datasets[0].data = S.training.trainData.slice();
    CHARTS.train.data.datasets[1].data = S.training.valData.slice();
    CHARTS.liveLR.data.datasets[0].data = S.training.lrData.slice();
    CHARTS.train.update('none');
    CHARTS.liveLR.update('none');

    document.getElementById('trainingStats').innerHTML = `
      <div class="stat-pill"><div class="sv">E${e}</div><div class="sk">Epoch</div></div>
      <div class="stat-pill"><div class="sv">${trainL.toFixed(4)}</div><div class="sk">Train Loss</div></div>
      <div class="stat-pill"><div class="sv">${lr.toExponential(2)}</div><div class="sk">LR</div></div>`;

    S.training.epoch++;
  }

  function trainingStart() {
    if (S.training.running) return;
    if (S.training.epoch > S.params.epochs) trainingReset();
    S.training.running = true;
    document.getElementById('startTraining').disabled = true;
    document.getElementById('pauseTraining').disabled = false;
    S.training.timer = setInterval(trainingStep, 60);
  }

  function trainingPause() {
    if (S.training.timer) { clearInterval(S.training.timer); S.training.timer = null; }
    S.training.running = false;
    document.getElementById('startTraining').disabled = false;
    document.getElementById('pauseTraining').disabled = true;
  }

  // ==========================================================
  // OPTIMIZATION SIMULATOR
  // ==========================================================
  let optCtx = null;
  let bgCanvas = null;

  function optLoss(x, y) { return x * x + 2 * y * y; }
  function optGrad(x, y) { return [2 * x, 4 * y]; }

  function buildBgCanvas(W, H) {
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const ctx = off.getContext('2d');
    const img = ctx.createImageData(W, H);
    const D = img.data;
    const maxL = 20;
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const wx = (px / W) * 6 - 3;
        const wy = (py / H) * 6 - 3;
        const l = Math.min(optLoss(wx, wy), maxL);
        const n = l / maxL;
        const hue = (1 - n) * 240;
        const [r, g, b] = hslToRgb(hue / 360, 0.75, 0.4 + n * 0.2);
        const idx = (py * W + px) * 4;
        D[idx] = r; D[idx + 1] = g; D[idx + 2] = b; D[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    [1, 2, 4, 7, 12].forEach(level => {
      ctx.beginPath();
      for (let angle = 0; angle <= Math.PI * 2; angle += 0.05) {
        const rx = Math.sqrt(level);
        const ry = Math.sqrt(level / 2);
        const sx = (W / 2) + (Math.cos(angle) * rx * W / 6);
        const sy = (H / 2) + (Math.sin(angle) * ry * H / 6);
        angle === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    });
    return off;
  }

  function worldToCanvas(wx, wy, W, H) {
    return [(wx + 3) / 6 * W, (wy + 3) / 6 * H];
  }

  function initOptCanvas() {
    const canvas = document.getElementById('optCanvas');
    const parent = canvas.parentElement;
    const size = Math.min(parent.offsetWidth || 380, 380);
    canvas.width = size;
    canvas.height = size;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    optCtx = canvas.getContext('2d');
    bgCanvas = buildBgCanvas(size, size);
  }

  function drawOptFrame() {
    if (!optCtx || !bgCanvas) return;
    const W = optCtx.canvas.width, H = optCtx.canvas.height;
    optCtx.drawImage(bgCanvas, 0, 0);

    if (S.opt.trail.length > 1) {
      optCtx.strokeStyle = 'rgba(255,255,255,0.6)';
      optCtx.lineWidth = 2;
      optCtx.beginPath();
      S.opt.trail.forEach((pt, i) => {
        const [cx, cy] = worldToCanvas(pt.x, pt.y, W, H);
        i === 0 ? optCtx.moveTo(cx, cy) : optCtx.lineTo(cx, cy);
      });
      optCtx.stroke();
    }

    const [bx, by] = worldToCanvas(S.opt.x, S.opt.y, W, H);
    optCtx.beginPath();
    optCtx.arc(bx, by, 8, 0, Math.PI * 2);
    optCtx.fillStyle = '#fff';
    optCtx.fill();
    optCtx.strokeStyle = '#f97316';
    optCtx.lineWidth = 3;
    optCtx.stroke();

    const [mx, my] = worldToCanvas(0, 0, W, H);
    optCtx.beginPath();
    optCtx.arc(mx, my, 5, 0, Math.PI * 2);
    optCtx.fillStyle = '#22c55e';
    optCtx.fill();

    optCtx.fillStyle = 'rgba(255,255,255,0.8)';
    optCtx.font = '11px monospace';
    optCtx.fillText('min', mx + 7, my + 4);
  }

  function optStep() {
    const e = S.opt.epoch;
    const ep = S.params.epochs;
    if (e > ep) { optPause(); return; }

    const lr = getLR(S.scheduler, e, S.params);
    const [gx, gy] = optGrad(S.opt.x, S.opt.y);
    S.opt.trail.push({ x: S.opt.x, y: S.opt.y });
    if (S.opt.trail.length > 200) S.opt.trail.shift();
    S.opt.x -= lr * gx;
    S.opt.y -= lr * gy;

    S.opt.x = Math.max(-3, Math.min(3, S.opt.x));
    S.opt.y = Math.max(-3, Math.min(3, S.opt.y));

    const loss = optLoss(S.opt.x, S.opt.y);
    S.opt.lossData.push({ x: e, y: loss });
    CHARTS.optLoss.data.datasets[0].data = S.opt.lossData.slice();
    CHARTS.optLoss.update('none');

    drawOptFrame();
    document.getElementById('optStats').innerHTML = `
      <div class="opt-stat">Epoch <span>${e}</span></div>
      <div class="opt-stat">LR <span>${lr.toExponential(2)}</span></div>
      <div class="opt-stat">Loss <span>${loss.toFixed(4)}</span></div>
      <div class="opt-stat">Pos <span>(${S.opt.x.toFixed(2)}, ${S.opt.y.toFixed(2)})</span></div>`;

    S.opt.epoch++;
  }

  function optStart() {
    if (S.opt.running) return;
    if (!optCtx) initOptCanvas();
    drawOptFrame();
    S.opt.running = true;
    document.getElementById('startOpt').disabled = true;
    document.getElementById('pauseOpt').disabled = false;
    S.opt.timer = setInterval(optStep, 120);
  }

  function optPause() {
    if (S.opt.timer) { clearInterval(S.opt.timer); S.opt.timer = null; }
    S.opt.running = false;
    document.getElementById('startOpt').disabled = false;
    document.getElementById('pauseOpt').disabled = true;
  }

  function optReset() {
    optPause();
    S.opt.epoch = 0; S.opt.x = 2.5; S.opt.y = 1.8;
    S.opt.trail = []; S.opt.lossData = [];
    CHARTS.optLoss.data.datasets[0].data = [];
    CHARTS.optLoss.update();
    document.getElementById('optStats').innerHTML = '';
    if (!optCtx) initOptCanvas();
    drawOptFrame();
  }

  // ==========================================================
  // ==========================================================
  // EXPORT UTILITIES
  // ==========================================================
  function exportPNG() {
    const url = CHARTS.lr.toBase64Image();
    const a = document.createElement('a');
    a.href = url; a.download = `${S.scheduler}_lr_schedule.png`; a.click();
  }

  function exportJSON() {
    const cfg = { scheduler: S.scheduler, params: { ...S.params }, generated: new Date().toISOString() };
    downloadFile(JSON.stringify(cfg, null, 2), `${S.scheduler}_config.json`, 'application/json');
  }

  function exportCSV() {
    const data = genData(S.scheduler, S.params);
    const rows = ['epoch,learning_rate', ...data.map(d => `${d.x},${d.y.toExponential(6)}`)];
    downloadFile(rows.join('\n'), `${S.scheduler}_schedule.csv`, 'text/csv');
  }

  function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }

  // ==========================================================
  // THEME
  // ==========================================================
  function toggleTheme() {
    S.theme = S.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', S.theme);
    document.getElementById('themeToggle').textContent = S.theme === 'dark' ? '🌙' : '☀️';
    updateChartThemes();
  }

  function updateChartThemes() {
    const tc = getThemeColors();
    Object.values(CHARTS).forEach(chart => {
      if (!chart) return;
      const opts = chart.options;
      if (opts.scales) {
        ['x', 'y'].forEach(ax => {
          if (opts.scales[ax]) {
            if (opts.scales[ax].grid)  opts.scales[ax].grid.color = tc.grid;
            if (opts.scales[ax].ticks) opts.scales[ax].ticks.color = tc.tick;
            if (opts.scales[ax].title) opts.scales[ax].title.color = tc.tick;
          }
        });
      }
      if (opts.plugins?.legend?.labels) opts.plugins.legend.labels.color = tc.tick;
      chart.update('none');
    });
    if (optCtx && bgCanvas) { bgCanvas = buildBgCanvas(optCtx.canvas.width, optCtx.canvas.height); drawOptFrame(); }
  }

  // ==========================================================
  // TABS
  // ==========================================================
  function switchTab(name) {
    S.activeTab = name;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
    if (name === 'comparison') { updateComparisonChart(); }
    if (name === 'optimization' && !optCtx) { setTimeout(initOptCanvas, 50); }
  }

  // ==========================================================
  // SCHEDULER SELECTION
  // ==========================================================
  function onSchedulerChange(name) {
    S.scheduler = name;
    renderControls();
    renderFormula();
    renderExplanation();
    updateMainChart();
    if (S.activeTab === 'comparison') updateComparisonChart();
  }

  // ==========================================================
  // UTILITIES
  // ==========================================================
  function hexAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) { r = g = b = l; }
    else {
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1; if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      };
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  // ==========================================================
  // SIDEBAR COLLAPSE
  // ==========================================================
  function initSidebarToggle() {
    const sidebar = document.getElementById('sidebar');
    const btn     = document.getElementById('sidebarToggle');
    btn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      btn.title = sidebar.classList.contains('collapsed') ? 'Expand sidebar' : 'Collapse sidebar';
    });
  }

  // ==========================================================
  // INIT
  // ==========================================================
  function init() {
    initCharts();
    renderControls();
    renderFormula();
    renderExplanation();
    updateMainChart();
    initComparisonUI();
    initSidebarToggle();

    // Scheduler dropdown
    document.getElementById('schedulerSelect').addEventListener('change', e => onSchedulerChange(e.target.value));

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Theme
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Presets
    document.querySelectorAll('.btn-preset').forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = PRESETS[btn.dataset.preset];
        if (!preset) return;
        S.scheduler = preset.sched;
        document.getElementById('schedulerSelect').value = preset.sched;
        Object.assign(S.params, preset.params);
        renderControls();
        renderFormula();
        renderExplanation();
        updateMainChart();
      });
    });

    // Export
    document.getElementById('exportPNG').addEventListener('click', exportPNG);
    document.getElementById('exportJSON').addEventListener('click', exportJSON);
    document.getElementById('exportCSV').addEventListener('click', exportCSV);
    document.getElementById('resetZoom').addEventListener('click', () => CHARTS.lr.resetZoom && CHARTS.lr.resetZoom());
    document.getElementById('resetZoomCmp').addEventListener('click', () => CHARTS.cmp.resetZoom && CHARTS.cmp.resetZoom());
    document.getElementById('resetZoomTrain').addEventListener('click', () => CHARTS.train.resetZoom && CHARTS.train.resetZoom());
    document.getElementById('resetZoomLiveLR').addEventListener('click', () => CHARTS.liveLR.resetZoom && CHARTS.liveLR.resetZoom());

    // Training sim
    document.getElementById('startTraining').addEventListener('click', trainingStart);
    document.getElementById('pauseTraining').addEventListener('click', trainingPause);
    document.getElementById('resetTraining').addEventListener('click', trainingReset);

    // Optimization
    document.getElementById('startOpt').addEventListener('click', optStart);
    document.getElementById('pauseOpt').addEventListener('click', optPause);
    document.getElementById('resetOpt').addEventListener('click', optReset);
    document.getElementById('resetZoomOptLoss').addEventListener('click', () => CHARTS.optLoss.resetZoom && CHARTS.optLoss.resetZoom());

  }

  document.addEventListener('DOMContentLoaded', init);

})();
