import type { TokenStep } from "../types";

interface Props {
  step: TokenStep | null;
}

function Bar({ label, value, tone }: { label: string; value: number; tone: "draft" | "target" }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className={`prob-row tone-${tone}`}>
      <div className="prob-label">
        <span>{label}</span>
        <span className="prob-val">{(value * 100).toFixed(2)}%</span>
      </div>
      <div className="prob-track" aria-hidden>
        <div className="prob-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function ProbCompare({ step }: Props) {
  if (!step) return null;

  const ratioLabel = Number.isFinite(step.ratio) ? step.ratio.toFixed(3) : "∞";

  return (
    <div className="prob-panel">
      <header className="prob-head">
        <div>
          <p className="eyebrow">Token-level distributions</p>
          <h3>
            <code>{step.display}</code>
            <span className={`pill pill-${step.outcome}`}>{step.outcome}</span>
          </h3>
        </div>
        <dl className="prob-stats">
          <div>
            <dt>q / p</dt>
            <dd>{ratioLabel}</dd>
          </div>
          <div>
            <dt>P(accept)</dt>
            <dd>{(step.acceptProb * 100).toFixed(1)}%</dd>
          </div>
          <div>
            <dt>roll</dt>
            <dd>{step.roll.toFixed(3)}</dd>
          </div>
        </dl>
      </header>

      {step.outcome !== "bonus" && (
        <div className="prob-bars">
          <Bar label="Draft p(x)" value={step.draftProb} tone="draft" />
          <Bar label="Target q(x)" value={step.targetProb} tone="target" />
        </div>
      )}

      {step.reason && <p className="prob-reason">{step.reason}</p>}
      {step.structuralHint && step.outcome === "accepted" && (
        <p className="prob-hint">{step.structuralHint}</p>
      )}
    </div>
  );
}
