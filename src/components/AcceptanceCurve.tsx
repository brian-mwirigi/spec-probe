import type { ProbeRun } from "../types";

interface Props {
  run: ProbeRun;
}

export function AcceptanceCurve({ run }: Props) {
  const draftSteps = run.steps.filter((s) => s.outcome !== "bonus");
  const width = 420;
  const height = 140;
  const pad = { t: 16, r: 12, b: 28, l: 36 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  if (draftSteps.length === 0) return null;

  const points = draftSteps.map((s, i) => {
    const x = pad.l + (i / Math.max(1, draftSteps.length - 1)) * innerW;
    const y = pad.t + (1 - s.acceptProb) * innerH;
    return { x, y, s };
  });

  const poly = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div className="curve-panel">
      <header>
        <p className="eyebrow">Acceptance by draft position</p>
        <h3>P(accept) along the speculative block</h3>
      </header>
      <svg
        className="curve-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Acceptance probability by draft token index"
      >
        <line
          x1={pad.l}
          y1={pad.t}
          x2={pad.l}
          y2={pad.t + innerH}
          className="axis"
        />
        <line
          x1={pad.l}
          y1={pad.t + innerH}
          x2={pad.l + innerW}
          y2={pad.t + innerH}
          className="axis"
        />
        <text x={4} y={pad.t + 4} className="axis-label">
          1
        </text>
        <text x={4} y={pad.t + innerH} className="axis-label">
          0
        </text>
        <polyline points={poly} className="curve-line" fill="none" />
        {points.map((p) => (
          <circle
            key={p.s.index}
            cx={p.x}
            cy={p.y}
            r={p.s.outcome === "rejected" && !p.s.reason?.startsWith("Unverified") ? 5.5 : 3.5}
            className={`curve-dot is-${p.s.outcome} ${p.s.reason?.startsWith("Unverified") ? "is-unverified" : ""}`}
          />
        ))}
        {draftSteps.map((s, i) => {
          const x = pad.l + (i / Math.max(1, draftSteps.length - 1)) * innerW;
          return (
            <text key={s.index} x={x} y={height - 8} textAnchor="middle" className="axis-label">
              {s.index}
            </text>
          );
        })}
        {run.rejectedAt !== null && (
          <line
            x1={pad.l + (run.rejectedAt / Math.max(1, draftSteps.length - 1)) * innerW}
            y1={pad.t}
            x2={pad.l + (run.rejectedAt / Math.max(1, draftSteps.length - 1)) * innerW}
            y2={pad.t + innerH}
            className="reject-guide"
          />
        )}
      </svg>
    </div>
  );
}
