// Shared drawing helpers for the NeuroDetect film. Everything is a pure function of (seed, t).
window.ND = (function () {
  const C = { cyan: "#22D3EE", violet: "#A78BFA", amber: "#F5B544", ink: "#F4F7FB", muted: "#8A97B5", grid: "#12203D" };

  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Side-view brain silhouette in unit space (x -1..1 front→back, y -0.7..0.9 top→bottom).
  function inBrain(x, y) {
    const cer = (x * x) / 1.0 + ((y + 0.02) * (y + 0.02)) / (0.66 * 0.66) < 1 && y < 0.36;
    const temporal = ((x + 0.15) * (x + 0.15)) / (0.62 * 0.62) + ((y - 0.3) * (y - 0.3)) / (0.2 * 0.2) < 1;
    const cbl = ((x - 0.55) * (x - 0.55)) / (0.32 * 0.32) + ((y - 0.48) * (y - 0.48)) / (0.17 * 0.17) < 1;
    const stem = x > 0.12 && x < 0.32 && y > 0.3 && y < 0.82 - (x - 0.12) * 0.6;
    return cer || temporal || cbl || stem;
  }

  const cache = {};
  function brain(n, seed) {
    const key = n + ":" + seed;
    if (cache[key]) return cache[key];
    const r = rng(seed), pts = [];
    while (pts.length < n) {
      const x = r() * 2 - 1, y = r() * 1.6 - 0.7;
      if (inBrain(x, y)) pts.push({ x, y, ph: r() * 6.283, s: 0.6 + r() * 1.4, hue: r() });
    }
    const edges = [];
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        if (dx * dx + dy * dy < 0.0042) edges.push([i, j, r()]);
      }
    return (cache[key] = { pts, edges });
  }

  // Draw the point-cloud brain. reveal 0..1 grows points in from the front; pulse adds travelling sparks.
  function drawBrain(ctx, cx, cy, size, t, o) {
    o = o || {};
    const { pts, edges } = brain(o.n || 700, o.seed || 7);
    const reveal = o.reveal == null ? 1 : o.reveal, alpha = o.alpha == null ? 1 : o.alpha;
    const P = (p) => [cx + p.x * size, cy + p.y * size];
    const vis = (p) => Math.min(1, Math.max(0, (reveal * 2.4 - (p.x + 1)) * 2));
    ctx.save();
    ctx.lineWidth = 1;
    for (const [i, j, k] of edges) {
      const a = Math.min(vis(pts[i]), vis(pts[j])) * alpha * 0.16;
      if (a <= 0.01) continue;
      const [x1, y1] = P(pts[i]), [x2, y2] = P(pts[j]);
      ctx.strokeStyle = k < 0.25 ? `rgba(167,139,250,${a})` : `rgba(34,211,238,${a})`;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    for (const p of pts) {
      const v = vis(p); if (v <= 0) continue;
      const tw = 0.55 + 0.45 * Math.sin(t * 2.2 + p.ph);
      const [x, y] = P(p);
      ctx.fillStyle = p.hue < 0.22 ? `rgba(167,139,250,${v * alpha * tw})` : `rgba(160,236,250,${v * alpha * tw})`;
      ctx.beginPath(); ctx.arc(x, y, p.s * (size / 420), 0, 6.283); ctx.fill();
    }
    if (o.pulse) {
      for (let e = 0; e < edges.length; e += 9) {
        const [i, j, k] = edges[e];
        const ph = (t * 0.9 + k * 7) % 1;
        const a = Math.sin(ph * Math.PI) * alpha * o.pulse * Math.min(vis(pts[i]), vis(pts[j]));
        if (a < 0.05) continue;
        const [x1, y1] = P(pts[i]), [x2, y2] = P(pts[j]);
        const x = x1 + (x2 - x1) * ph, y = y1 + (y2 - y1) * ph;
        ctx.shadowColor = C.cyan; ctx.shadowBlur = 12;
        ctx.fillStyle = `rgba(230,252,255,${a})`;
        ctx.beginPath(); ctx.arc(x, y, 2.2 * (size / 420), 0, 6.283); ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
    ctx.restore();
  }

  // EEG-like signal value at time s (seconds of signal). kind: "alpha" | "theta" | "raw".
  function eeg(s, seed, kind) {
    const r = rng(seed); const ph = [r(), r(), r(), r(), r(), r()].map((v) => v * 6.283);
    const alpha = kind === "theta" ? 0.35 : 1, theta = kind === "theta" ? 1.1 : 0.35;
    let v = alpha * Math.sin(6.283 * 10 * s + ph[0]) * (0.7 + 0.3 * Math.sin(6.283 * 0.4 * s + ph[1]))
      + theta * Math.sin(6.283 * 6 * s + ph[2]) * 0.8
      + 0.25 * Math.sin(6.283 * 2.2 * s + ph[3])
      + 0.12 * Math.sin(6.283 * 21 * s + ph[4]);
    if (kind === "raw") v += 0.55 * Math.sin(6.283 * 50 * s + ph[5]) + 0.4 * Math.sin(6.283 * 0.3 * s);
    return v;
  }

  // Draw a trace across [x, x+w]; window = seconds shown, t = current playhead in signal seconds.
  function drawTrace(ctx, x, y, w, amp, t, o) {
    o = o || {};
    const win = o.window || 2, n = Math.floor(w / 2), draw = o.draw == null ? 1 : o.draw;
    ctx.save();
    ctx.strokeStyle = o.color || C.cyan; ctx.lineWidth = o.width || 2.5;
    ctx.globalAlpha = o.alpha == null ? 1 : o.alpha;
    if (o.glow) { ctx.shadowColor = o.color || C.cyan; ctx.shadowBlur = o.glow; }
    ctx.beginPath();
    for (let i = 0; i <= n * draw; i++) {
      const s = t + (i / n) * win;
      const yy = y - eeg(s, o.seed || 3, o.kind || "alpha") * amp;
      i ? ctx.lineTo(x + (i / n) * w, yy) : ctx.moveTo(x, yy);
    }
    ctx.stroke(); ctx.restore();
  }

  // 10–20 system, 19 electrodes, top view in unit circle (nose up).
  const SITES = {
    Fp1: [-0.31, -0.95], Fp2: [0.31, -0.95], F7: [-0.81, -0.59], F3: [-0.4, -0.52], Fz: [0, -0.5], F4: [0.4, -0.52], F8: [0.81, -0.59],
    T3: [-1, 0], C3: [-0.5, 0], Cz: [0, 0], C4: [0.5, 0], T4: [1, 0],
    T5: [-0.81, 0.59], P3: [-0.4, 0.52], Pz: [0, 0.5], P4: [0.4, 0.52], T6: [0.81, 0.59], O1: [-0.31, 0.95], O2: [0.31, 0.95],
  };

  // Build an SVG head into `svg` (an <svg> element, viewBox -1.4 -1.4 2.8 2.8). Returns {name: <g>}.
  function head(svg, prefix) {
    const NS = "http://www.w3.org/2000/svg", mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); return e; };
    svg.setAttribute("viewBox", "-1.45 -1.45 2.9 2.9");
    svg.appendChild(mk("path", { d: "M -0.14 -1.17 L 0 -1.36 L 0.14 -1.17", fill: "none", stroke: "#2A3A63", "stroke-width": 0.025 }));
    svg.appendChild(mk("ellipse", { cx: -1.2, cy: 0, rx: 0.07, ry: 0.2, fill: "none", stroke: "#2A3A63", "stroke-width": 0.025 }));
    svg.appendChild(mk("ellipse", { cx: 1.2, cy: 0, rx: 0.07, ry: 0.2, fill: "none", stroke: "#2A3A63", "stroke-width": 0.025 }));
    svg.appendChild(mk("circle", { cx: 0, cy: 0, r: 1.17, fill: "#0B1328", stroke: "#2A3A63", "stroke-width": 0.025 }));
    const out = {};
    for (const n in SITES) {
      const [x, y] = SITES[n];
      const g = mk("g", { id: prefix + "-" + n });
      g.appendChild(mk("circle", { cx: x, cy: y, r: 0.1, fill: "#16213F", stroke: "#3A4C7A", "stroke-width": 0.015 }));
      const tx = mk("text", { x: x, y: y + 0.035, "text-anchor": "middle", "font-size": 0.085, "font-family": "IBM Plex Mono", fill: "#8A97B5" });
      tx.textContent = n; g.appendChild(tx);
      svg.appendChild(g); out[n] = g;
    }
    return out;
  }

  function fitCanvas(cv) { cv.width = 1920; cv.height = 1080; return cv.getContext("2d"); }

  return { C, rng, brain, drawBrain, eeg, drawTrace, SITES, head, fitCanvas };
})();
