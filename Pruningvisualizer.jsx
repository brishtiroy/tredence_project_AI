import { useState, useEffect, useRef, useCallback } from "react";

// ─── Palette & constants ──────────────────────────────────────────────────────
const COLORS = {
  bg:       "#0a0c10",
  surface:  "#0f1318",
  panel:    "#141920",
  border:   "#1e2530",
  accent:   "#00e5ff",
  accent2:  "#ff3d71",
  accent3:  "#39ff14",
  muted:    "#3a4555",
  text:     "#e2e8f0",
  textDim:  "#6b7a90",
};

// ─── Simulation helpers ────────────────────────────────────────────────────────
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
}

function generateGates(lambda, totalWeights = 2000, seed = 42) {
  const rng = seededRandom(seed + lambda * 10000);
  // Higher lambda → more gates pushed to zero
  const sparsityTarget = Math.min(0.97, 0.3 + lambda * 8);
  const gates = [];
  for (let i = 0; i < totalWeights; i++) {
    const r = rng();
    if (r < sparsityTarget) {
      // pruned gate: near-zero
      gates.push(rng() * 0.04);
    } else {
      // active gate: spread across (0.1, 1.0)
      gates.push(0.1 + rng() * 0.88);
    }
  }
  return gates;
}

function simulateTraining(lambda, epochs = 40) {
  const rng = seededRandom(Math.round(lambda * 100000));
  const baseAcc = 0.86 - lambda * 5.5;
  const finalSparsity = Math.min(0.97, 0.3 + lambda * 8);
  const history = { val_acc: [], sparsity: [], ce_loss: [], sp_loss: [], total_loss: [] };

  for (let ep = 0; ep < epochs; ep++) {
    const t = ep / epochs;
    const noise = (rng() - 0.5) * 0.015;
    const acc = baseAcc * (1 - Math.exp(-4 * t)) + noise;
    const sp  = finalSparsity * (1 - Math.exp(-3 * t)) + (rng() - 0.5) * 0.02;
    const ce  = 2.3 * Math.exp(-3 * t) + 0.2 + (rng() - 0.5) * 0.05;
    const spl = 5000 * (1 - sp);
    history.val_acc.push(Math.max(0.1, Math.min(0.98, acc)));
    history.sparsity.push(Math.max(0, Math.min(0.99, sp)));
    history.ce_loss.push(Math.max(0.1, ce));
    history.sp_loss.push(spl);
    history.total_loss.push(ce + lambda * spl);
  }
  return history;
}

const LAMBDA_CONFIGS = [
  { value: 1e-4, label: "λ = 0.0001", tag: "Low",    color: COLORS.accent },
  { value: 1e-3, label: "λ = 0.001",  tag: "Medium", color: COLORS.accent3 },
  { value: 1e-2, label: "λ = 0.01",   tag: "High",   color: COLORS.accent2 },
];

// ─── Mini SVG line chart ───────────────────────────────────────────────────────
function SparkLine({ data, color, height = 50, width = 200 }) {
  if (!data || data.length < 2) return null;
  const mn = Math.min(...data), mx = Math.max(...data);
  const range = mx - mn || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - mn) / range) * height * 0.9 - height * 0.05;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id={`g${color.replace("#","")}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon
        points={`0,${height} ${pts} ${width},${height}`}
        fill={`url(#g${color.replace("#","")})`}
      />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5"
                strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// ─── Gate histogram bars ───────────────────────────────────────────────────────
function GateHistogram({ gates, color, threshold = 0.05 }) {
  const bins = 40;
  const counts = new Array(bins).fill(0);
  gates.forEach(g => {
    const b = Math.min(bins - 1, Math.floor(g * bins));
    counts[b]++;
  });
  const maxCount = Math.max(...counts, 1);
  const pruned = gates.filter(g => g < threshold).length;
  const pct = ((pruned / gates.length) * 100).toFixed(1);

  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 72 }}>
        {counts.map((c, i) => {
          const isPruned = i / bins < threshold;
          const h = (c / maxCount) * 68;
          return (
            <div
              key={i}
              title={`Gate range: ${(i/bins).toFixed(2)}–${((i+1)/bins).toFixed(2)}\nCount: ${c}`}
              style={{
                flex: 1, height: h, minHeight: c > 0 ? 2 : 0,
                background: isPruned ? COLORS.accent2 : color,
                borderRadius: "1px 1px 0 0",
                opacity: isPruned ? 0.9 : 0.7,
                transition: "height 0.6s cubic-bezier(.4,0,.2,1)",
                alignSelf: "flex-end",
              }}
            />
          );
        })}
      </div>
      <div style={{
        display: "flex", justifyContent: "space-between",
        marginTop: 6, fontSize: 10, color: COLORS.textDim,
      }}>
        <span>0.0</span>
        <span style={{ color: COLORS.accent2, fontWeight: 700 }}>
          {pct}% pruned
        </span>
        <span>1.0</span>
      </div>
    </div>
  );
}

// ─── Animated neural net SVG ───────────────────────────────────────────────────
function NeuralNetViz({ activeLambda, isTraining, epoch }) {
  const layers = [3, 6, 5, 4, 3, 2];
  const W = 260, H = 200;
  const layerX = layers.map((_, i) => 20 + (i / (layers.length - 1)) * (W - 40));
  const sparsity = isTraining
    ? Math.min(0.9, 0.3 + activeLambda * 8) * (epoch / 40)
    : Math.min(0.9, 0.3 + activeLambda * 8);

  const rng = seededRandom(42 + activeLambda * 999);
  const nodePositions = layers.map((count, li) => {
    const xs = layerX[li];
    return Array.from({ length: count }, (_, ni) => ({
      x: xs,
      y: 20 + (ni / (count - 1 || 1)) * (H - 40),
    }));
  });

  const edges = [];
  for (let li = 0; li < layers.length - 1; li++) {
    for (let ni = 0; ni < layers[li]; ni++) {
      for (let nj = 0; nj < layers[li + 1]; nj++) {
        const pruned = rng() < sparsity;
        edges.push({
          x1: nodePositions[li][ni].x,
          y1: nodePositions[li][ni].y,
          x2: nodePositions[li + 1][nj].x,
          y2: nodePositions[li + 1][nj].y,
          pruned,
        });
      }
    }
  }

  return (
    <svg width={W} height={H} style={{ display: "block", margin: "0 auto" }}>
      {edges.map((e, i) => (
        <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
          stroke={e.pruned ? COLORS.accent2 : COLORS.accent}
          strokeWidth={e.pruned ? 0.4 : 0.8}
          strokeOpacity={e.pruned ? 0.15 : 0.35}
        />
      ))}
      {nodePositions.flat().map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={5}
          fill={COLORS.panel} stroke={COLORS.accent} strokeWidth={1.2}
          style={{ filter: `drop-shadow(0 0 3px ${COLORS.accent}55)` }}
        />
      ))}
    </svg>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedLambda, setSelectedLambda] = useState(1e-3);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(40);
  const [animEpoch, setAnimEpoch] = useState(40);
  const [activeTab, setActiveTab] = useState("overview");
  const [hoveredLambda, setHoveredLambda] = useState(null);
  const intervalRef = useRef(null);

  const allData = LAMBDA_CONFIGS.reduce((acc, cfg) => {
    acc[cfg.value] = {
      history: simulateTraining(cfg.value),
      gates:   generateGates(cfg.value),
    };
    return acc;
  }, {});

  const current = allData[selectedLambda];
  const cfg = LAMBDA_CONFIGS.find(c => c.value === selectedLambda);

  const startTraining = useCallback(() => {
    setIsTraining(true);
    setAnimEpoch(0);
    let ep = 0;
    intervalRef.current = setInterval(() => {
      ep++;
      setAnimEpoch(ep);
      if (ep >= 40) {
        clearInterval(intervalRef.current);
        setIsTraining(false);
        setAnimEpoch(40);
      }
    }, 60);
  }, []);

  useEffect(() => () => clearInterval(intervalRef.current), []);

  const displayEpoch = isTraining ? animEpoch : 40;
  const displayHistory = {
    val_acc:    current.history.val_acc.slice(0, displayEpoch),
    sparsity:   current.history.sparsity.slice(0, displayEpoch),
    ce_loss:    current.history.ce_loss.slice(0, displayEpoch),
    total_loss: current.history.total_loss.slice(0, displayEpoch),
  };

  const finalAcc     = (current.history.val_acc[displayEpoch - 1] || 0) * 100;
  const finalSp      = (current.history.sparsity[displayEpoch - 1] || 0) * 100;
  const pruned       = Math.round((finalSp / 100) * current.gates.length);

  const tabs = [
    { id: "overview",  label: "Overview" },
    { id: "training",  label: "Training Curves" },
    { id: "gates",     label: "Gate Distribution" },
    { id: "network",   label: "Network View" },
  ];

  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      color: COLORS.text,
      padding: "24px 20px",
      boxSizing: "border-box",
    }}>
      {/* ── Header ── */}
      <div style={{ marginBottom: 28, borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <span style={{
            fontSize: 11, letterSpacing: 4, color: COLORS.accent,
            textTransform: "uppercase", fontWeight: 700,
          }}>
            TREDENCE · 2025 COHORT
          </span>
          <span style={{ color: COLORS.muted, fontSize: 11 }}>▸</span>
          <span style={{ color: COLORS.textDim, fontSize: 11, letterSpacing: 2 }}>
            AI ENGINEERING CASE STUDY
          </span>
        </div>
        <h1 style={{
          margin: "8px 0 4px",
          fontSize: "clamp(18px, 3vw, 28px)",
          fontWeight: 700,
          letterSpacing: -0.5,
          color: "#fff",
        }}>
          Self-Pruning Neural Network
        </h1>
        <p style={{ margin: 0, fontSize: 12, color: COLORS.textDim, letterSpacing: 0.3 }}>
          Dynamic weight pruning via learnable sigmoid gates + L1 sparsity regularisation
        </p>
      </div>

      {/* ── Lambda selector ── */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: COLORS.textDim, marginBottom: 10, textTransform: "uppercase" }}>
          Select Lambda (λ) — Sparsity Trade-off
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {LAMBDA_CONFIGS.map(c => (
            <button key={c.value}
              onClick={() => { setSelectedLambda(c.value); setAnimEpoch(40); setIsTraining(false); }}
              onMouseEnter={() => setHoveredLambda(c.value)}
              onMouseLeave={() => setHoveredLambda(null)}
              style={{
                padding: "8px 16px",
                background: selectedLambda === c.value ? `${c.color}18` : "transparent",
                border: `1px solid ${selectedLambda === c.value ? c.color : COLORS.border}`,
                color: selectedLambda === c.value ? c.color : COLORS.textDim,
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 11,
                fontFamily: "inherit",
                letterSpacing: 1,
                transition: "all 0.15s",
                display: "flex", alignItems: "center", gap: 8,
              }}>
              <span style={{
                display: "inline-block", width: 6, height: 6, borderRadius: "50%",
                background: c.color, opacity: selectedLambda === c.value ? 1 : 0.4,
              }} />
              {c.label}
              <span style={{
                fontSize: 9, letterSpacing: 1, padding: "1px 5px",
                background: `${c.color}22`, color: c.color,
                borderRadius: 2, textTransform: "uppercase",
              }}>{c.tag}</span>
            </button>
          ))}
          <button onClick={startTraining} disabled={isTraining}
            style={{
              padding: "8px 16px", marginLeft: "auto",
              background: isTraining ? `${COLORS.accent3}11` : `${COLORS.accent3}22`,
              border: `1px solid ${COLORS.accent3}`,
              color: COLORS.accent3, borderRadius: 4, cursor: isTraining ? "not-allowed" : "pointer",
              fontSize: 11, fontFamily: "inherit", letterSpacing: 1,
              display: "flex", alignItems: "center", gap: 6,
              transition: "all 0.15s", opacity: isTraining ? 0.6 : 1,
            }}>
            {isTraining ? (
              <>
                <span style={{
                  display: "inline-block", width: 8, height: 8, borderRadius: "50%",
                  background: COLORS.accent3,
                  animation: "pulse 0.8s ease-in-out infinite",
                }} />
                TRAINING... ep {animEpoch}/40
              </>
            ) : "▶ SIMULATE TRAINING"}
          </button>
        </div>
      </div>

      {/* ── Stat cards ── */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
        gap: 10, marginBottom: 24,
      }}>
        {[
          { label: "Test Accuracy",   value: `${finalAcc.toFixed(1)}%`, color: COLORS.accent },
          { label: "Sparsity Level",  value: `${finalSp.toFixed(1)}%`,  color: cfg.color },
          { label: "Pruned Weights",  value: `${pruned.toLocaleString()}`, color: COLORS.accent2 },
          { label: "Active Weights",  value: `${(current.gates.length - pruned).toLocaleString()}`, color: COLORS.accent3 },
          { label: "Epoch",           value: `${displayEpoch} / 40`,    color: COLORS.textDim },
          { label: "Lambda",          value: selectedLambda.toExponential(0), color: cfg.color },
        ].map(s => (
          <div key={s.label} style={{
            background: COLORS.panel,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6, padding: "12px 14px",
          }}>
            <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, textTransform: "uppercase", marginBottom: 6 }}>
              {s.label}
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: s.color, letterSpacing: -0.5 }}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      {/* ── Tabs ── */}
      <div style={{ display: "flex", gap: 0, marginBottom: 0, borderBottom: `1px solid ${COLORS.border}` }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            style={{
              padding: "10px 16px", background: "transparent",
              border: "none", borderBottom: `2px solid ${activeTab === t.id ? COLORS.accent : "transparent"}`,
              color: activeTab === t.id ? COLORS.accent : COLORS.textDim,
              cursor: "pointer", fontSize: 11, fontFamily: "inherit",
              letterSpacing: 1.5, textTransform: "uppercase",
              transition: "all 0.15s", marginBottom: -1,
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Tab content ── */}
      <div style={{
        background: COLORS.panel, border: `1px solid ${COLORS.border}`,
        borderTop: "none", borderRadius: "0 0 8px 8px", padding: 20,
      }}>

        {/* OVERVIEW TAB */}
        {activeTab === "overview" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* Accuracy sparkline */}
              <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 14 }}>
                <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 8, textTransform: "uppercase" }}>
                  Validation Accuracy
                </div>
                <SparkLine data={displayHistory.val_acc.map(v => v * 100)} color={COLORS.accent} />
              </div>
              {/* Sparsity sparkline */}
              <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 14 }}>
                <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 8, textTransform: "uppercase" }}>
                  Sparsity Growth
                </div>
                <SparkLine data={displayHistory.sparsity.map(v => v * 100)} color={cfg.color} />
              </div>
            </div>

            {/* Lambda comparison table */}
            <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 14 }}>
              <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 12, textTransform: "uppercase" }}>
                Lambda Comparison — Final Results
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr>
                    {["Lambda", "Test Accuracy", "Sparsity", "Pruned / Total", "Trade-off"].map(h => (
                      <th key={h} style={{
                        textAlign: "left", padding: "4px 8px",
                        color: COLORS.textDim, fontWeight: 500,
                        borderBottom: `1px solid ${COLORS.border}`,
                        letterSpacing: 1, fontSize: 10,
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {LAMBDA_CONFIGS.map(c => {
                    const d = allData[c.value];
                    const acc = (d.history.val_acc[39] * 100).toFixed(1);
                    const sp  = (d.history.sparsity[39] * 100).toFixed(1);
                    const pr  = Math.round(d.history.sparsity[39] * d.gates.length);
                    const tot = d.gates.length;
                    const isActive = c.value === selectedLambda;
                    return (
                      <tr key={c.value} style={{
                        background: isActive ? `${c.color}08` : "transparent",
                        cursor: "pointer",
                        transition: "background 0.15s",
                      }}
                        onClick={() => setSelectedLambda(c.value)}>
                        <td style={{ padding: "8px 8px", borderBottom: `1px solid ${COLORS.border}44` }}>
                          <span style={{ color: c.color, fontWeight: 700 }}>{c.label}</span>
                          <span style={{ marginLeft: 6, fontSize: 9, color: COLORS.textDim }}>{c.tag}</span>
                        </td>
                        <td style={{ padding: "8px 8px", borderBottom: `1px solid ${COLORS.border}44` }}>
                          <span style={{ color: COLORS.accent }}>{acc}%</span>
                        </td>
                        <td style={{ padding: "8px 8px", borderBottom: `1px solid ${COLORS.border}44` }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <div style={{
                              width: 60, height: 4, background: COLORS.muted, borderRadius: 2, overflow: "hidden",
                            }}>
                              <div style={{
                                width: `${sp}%`, height: "100%",
                                background: c.color, borderRadius: 2,
                                transition: "width 0.6s",
                              }} />
                            </div>
                            <span style={{ color: c.color }}>{sp}%</span>
                          </div>
                        </td>
                        <td style={{ padding: "8px 8px", borderBottom: `1px solid ${COLORS.border}44`, color: COLORS.textDim }}>
                          {pr.toLocaleString()} / {tot.toLocaleString()}
                        </td>
                        <td style={{ padding: "8px 8px", borderBottom: `1px solid ${COLORS.border}44` }}>
                          <span style={{ fontSize: 9, color: COLORS.textDim }}>
                            {c.tag === "Low" ? "High acc, less sparse" :
                             c.tag === "High" ? "High sparse, less acc" :
                             "Balanced"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TRAINING CURVES TAB */}
        {activeTab === "training" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {[
              { key: "val_acc",    label: "Validation Accuracy (%)", mul: 100, color: COLORS.accent },
              { key: "sparsity",   label: "Sparsity Level (%)",       mul: 100, color: cfg.color },
              { key: "ce_loss",    label: "Cross-Entropy Loss",       mul: 1,   color: "#facc15" },
              { key: "total_loss", label: "Total Loss (CE + λ·Sp)",   mul: 1,   color: COLORS.accent2 },
            ].map(({ key, label, mul, color }) => (
              <div key={key} style={{
                background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                borderRadius: 6, padding: 14,
              }}>
                <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 10, textTransform: "uppercase" }}>
                  {label}
                </div>
                <div style={{ position: "relative", height: 80 }}>
                  <svg width="100%" height="80" preserveAspectRatio="none">
                    {(() => {
                      const data = (displayHistory[key] || []).map(v => v * mul);
                      if (data.length < 2) return null;
                      const mn = Math.min(...data), mx = Math.max(...data);
                      const range = mx - mn || 1;
                      const w = 100, h = 80;
                      const pts = data.map((v, i) => {
                        const x = (i / (data.length - 1)) * w;
                        const y = h - ((v - mn) / range) * h * 0.85 - h * 0.07;
                        return `${x},${y}`;
                      }).join(" ");
                      return (
                        <>
                          <defs>
                            <linearGradient id={`tg${key}`} x1="0" x2="0" y1="0" y2="1">
                              <stop offset="0%" stopColor={color} stopOpacity="0.25" />
                              <stop offset="100%" stopColor={color} stopOpacity="0" />
                            </linearGradient>
                          </defs>
                          <polygon
                            points={`0,${h} ${pts} ${w},${h}`}
                            fill={`url(#tg${key})`}
                          />
                          <polyline points={pts} fill="none" stroke={color}
                            strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"
                            vectorEffect="non-scaling-stroke"
                          />
                        </>
                      );
                    })()}
                  </svg>
                </div>
                {/* All 3 lambdas faint overlay */}
                <div style={{ marginTop: 8, display: "flex", gap: 12 }}>
                  {LAMBDA_CONFIGS.map(c => {
                    const d = allData[c.value];
                    const vals = (d.history[key] || []).map(v => v * mul);
                    const last = vals[Math.min(displayEpoch - 1, vals.length - 1)] || 0;
                    return (
                      <span key={c.value} style={{
                        fontSize: 9, color: c.value === selectedLambda ? c.color : COLORS.muted,
                        display: "flex", alignItems: "center", gap: 4,
                      }}>
                        <span style={{ width: 8, height: 2, background: c.color, display: "inline-block", borderRadius: 1 }} />
                        {last.toFixed(key.includes("loss") ? 2 : 1)}{key === "val_acc" || key === "sparsity" ? "%" : ""}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* GATE DISTRIBUTION TAB */}
        {activeTab === "gates" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14 }}>
              {LAMBDA_CONFIGS.map(c => {
                const gates = allData[c.value].gates;
                const sp = (gates.filter(g => g < 0.05).length / gates.length * 100).toFixed(1);
                return (
                  <div key={c.value}
                    onClick={() => setSelectedLambda(c.value)}
                    style={{
                      background: COLORS.surface,
                      border: `1px solid ${c.value === selectedLambda ? c.color : COLORS.border}`,
                      borderRadius: 6, padding: 14, cursor: "pointer",
                      transition: "border-color 0.15s",
                    }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                      <span style={{ fontSize: 11, color: c.color, fontWeight: 700 }}>{c.label}</span>
                      <span style={{ fontSize: 9, color: COLORS.textDim }}>{c.tag}</span>
                    </div>
                    <GateHistogram gates={gates} color={c.color} />
                    <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                      <div style={{ fontSize: 9, color: COLORS.textDim }}>
                        Mean gate: <span style={{ color: c.color }}>
                          {(gates.reduce((a, b) => a + b, 0) / gates.length).toFixed(3)}
                        </span>
                      </div>
                      <div style={{ fontSize: 9, color: COLORS.textDim }}>
                        Pruned: <span style={{ color: COLORS.accent2 }}>{sp}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div style={{
              marginTop: 14, background: COLORS.surface,
              border: `1px solid ${COLORS.border}`, borderRadius: 6, padding: 14,
            }}>
              <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 8, textTransform: "uppercase" }}>
                Why the spike at 0 indicates success
              </div>
              <p style={{ margin: 0, fontSize: 11, color: COLORS.textDim, lineHeight: 1.7 }}>
                A successful self-pruning run produces a{" "}
                <span style={{ color: COLORS.accent }}>bimodal gate distribution</span>:
                a large spike near 0 (pruned weights) and a smaller cluster of active gates.
                The <span style={{ color: COLORS.accent2 }}>L1 penalty</span> imposes a constant
                gradient on every gate regardless of its magnitude — unlike L2, which shrinks
                the gradient toward zero, L1 keeps pushing until gates hit exactly 0.
              </p>
            </div>
          </div>
        )}

        {/* NETWORK VIEW TAB */}
        {activeTab === "network" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start" }}>
              <div style={{
                background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                borderRadius: 6, padding: 16,
              }}>
                <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 12, textTransform: "uppercase" }}>
                  Live Network — {cfg.label}
                </div>
                <NeuralNetViz activeLambda={selectedLambda} isTraining={isTraining} epoch={displayEpoch} />
                <div style={{ display: "flex", gap: 16, marginTop: 12, justifyContent: "center", fontSize: 10 }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 5, color: COLORS.textDim }}>
                    <span style={{ width: 16, height: 1.5, background: COLORS.accent, display: "inline-block" }} />
                    Active connection
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: 5, color: COLORS.textDim }}>
                    <span style={{ width: 16, height: 1.5, background: COLORS.accent2, display: "inline-block", opacity: 0.4 }} />
                    Pruned connection
                  </span>
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {/* Architecture breakdown */}
                <div style={{
                  background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                  borderRadius: 6, padding: 14,
                }}>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 12, textTransform: "uppercase" }}>
                    Layer Architecture
                  </div>
                  {[
                    { name: "Conv Backbone",    type: "non-prunable", params: "1.2M", color: COLORS.muted },
                    { name: "FC1  512 → 256",   type: "PrunableLinear", params: "131,072 gates", color: cfg.color },
                    { name: "FC2  256 → 128",   type: "PrunableLinear", params: "32,768 gates",  color: cfg.color },
                    { name: "FC3  128 → 10",    type: "PrunableLinear", params: "1,280 gates",   color: cfg.color },
                  ].map((l, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "8px 0",
                      borderBottom: i < 3 ? `1px solid ${COLORS.border}44` : "none",
                    }}>
                      <div style={{
                        width: 6, height: 6, borderRadius: "50%",
                        background: l.color, flexShrink: 0,
                      }} />
                      <div>
                        <div style={{ fontSize: 11, color: COLORS.text, fontWeight: 600 }}>{l.name}</div>
                        <div style={{ fontSize: 9, color: COLORS.textDim, marginTop: 1 }}>
                          {l.type} · {l.params}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Formula */}
                <div style={{
                  background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                  borderRadius: 6, padding: 14,
                }}>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: COLORS.textDim, marginBottom: 10, textTransform: "uppercase" }}>
                    Core Equations
                  </div>
                  {[
                    { label: "Forward pass",   eq: "out = (W ⊙ σ(G)) x + b", color: COLORS.accent },
                    { label: "Sparsity loss",  eq: "ℒsp = Σᵢⱼ σ(gᵢⱼ)", color: cfg.color },
                    { label: "Total loss",     eq: "ℒ = ℒCE + λ · ℒsp", color: "#facc15" },
                  ].map(f => (
                    <div key={f.label} style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 9, color: COLORS.textDim }}>{f.label}</div>
                      <div style={{
                        fontFamily: "monospace", fontSize: 11, color: f.color,
                        background: `${f.color}0c`, padding: "4px 8px",
                        borderRadius: 3, marginTop: 3, letterSpacing: 0.5,
                      }}>{f.eq}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div style={{
        marginTop: 20, paddingTop: 14, borderTop: `1px solid ${COLORS.border}`,
        display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8,
      }}>
        <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 2 }}>
          CIFAR-10 · PYTORCH · SELF-PRUNING VIA SIGMOID GATES
        </span>
        <span style={{ fontSize: 9, color: COLORS.muted, letterSpacing: 1 }}>
          <span style={{ color: COLORS.accent }}>▸</span>{" "}
          python self_pruning_network.py --epochs 40 --lambdas 1e-4 1e-3 1e-2
        </span>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        * { box-sizing: border-box; }
        button:hover { opacity: 0.85; }
      `}</style>
    </div>
  );
}