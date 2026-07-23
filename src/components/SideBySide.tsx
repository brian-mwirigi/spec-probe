import type { ProbeRun, StructuralFailure, TokenStep } from "../types";
import { TokenLane } from "./TokenLane";

interface Props {
  lanes: Array<{ strategy: string; run: ProbeRun }>;
  selected: Record<string, number>;
  onSelect: (strategy: string, index: number) => void;
  prompt?: string;
}

function verdict(lanes: Props["lanes"]): string {
  const ngram = lanes.find((l) => l.strategy === "ngram");
  const eagle = lanes.find((l) => l.strategy === "eagle" || l.strategy.startsWith("eagle"));
  if (ngram && eagle) {
    if (ngram.run.rejectedAt !== null && eagle.run.rejectedAt === null) {
      return "EAGLE clears the block. n-gram dies at the first draft step.";
    }
    if ((eagle.run.acceptanceRate ?? 0) > (ngram.run.acceptanceRate ?? 0) + 0.2) {
      return "EAGLE holds distributional alignment where n-gram collapses.";
    }
  }
  const best = [...lanes].sort((a, b) => b.run.acceptanceRate - a.run.acceptanceRate)[0];
  return `${best.strategy} leads at ${(best.run.acceptanceRate * 100).toFixed(0)}% acceptance on this prompt.`;
}

function MiniBars({ step }: { step: TokenStep | undefined }) {
  if (!step || step.outcome === "bonus") return null;
  const p = Math.min(100, step.draftProb * 100);
  const q = Math.min(100, step.targetProb * 100);
  return (
    <div className="sbs-minibars" aria-hidden>
      <div className="sbs-mini">
        <span>p</span>
        <i style={{ width: `${p}%` }} className="tone-p" />
      </div>
      <div className="sbs-mini">
        <span>q</span>
        <i style={{ width: `${q}%` }} className="tone-q" />
      </div>
    </div>
  );
}

export function SideBySide({ lanes, selected, onSelect, prompt }: Props) {
  if (lanes.length === 0) return null;

  const ordered = [...lanes].sort((a, b) => {
    const rank = (s: string) => (s === "ngram" ? 0 : s.startsWith("eagle") ? 1 : 2);
    return rank(a.strategy) - rank(b.strategy);
  });

  return (
    <section className="side-by-side" aria-label="Strategy side-by-side">
      <header className="sbs-hero">
        <p className="eyebrow">Artifact compare · same prompt · same residual math</p>
        <h2>EAGLE vs n-gram</h2>
        <p className="sbs-verdict">{verdict(ordered)}</p>
        {prompt && <pre className="sbs-prompt">{prompt}</pre>}
      </header>

      <div className={`sbs-grid cols-${Math.min(ordered.length, 3)}`}>
        {ordered.map(({ strategy, run }) => {
          const sel = selected[strategy] ?? (run.rejectedAt ?? 0);
          const step = run.steps[sel];
          const crush = strategy === "ngram" && run.rejectedAt !== null;
          const clear = (strategy === "eagle" || strategy.startsWith("eagle")) && run.rejectedAt === null;

          return (
            <article
              key={strategy}
              className={`sbs-panel ${crush ? "is-crush" : ""} ${clear ? "is-clear" : ""}`}
            >
              <header className="sbs-head">
                <h3>{strategy}</h3>
                <span className={run.acceptanceRate < 0.5 ? "warn" : "ok"}>
                  {(run.acceptanceRate * 100).toFixed(0)}% accept
                </span>
              </header>

              <p className="sbs-meta">
                {run.rejectedAt === null ? (
                  <>full block · bonus token recovered from target</>
                ) : (
                  <>
                    reject @ {run.rejectedAt}
                    {run.structuralFailures[0] ? ` · ${run.structuralFailures[0].label}` : ""}
                  </>
                )}
              </p>

              <TokenLane
                steps={run.steps}
                selected={sel}
                onSelect={(i) => onSelect(strategy, i)}
              />

              <MiniBars step={step} />

              {step && (
                <p className="sbs-step-note">
                  <code>{step.display}</code>
                  {" · "}
                  q/p={Number.isFinite(step.ratio) ? step.ratio.toFixed(2) : "∞"}
                  {" · "}
                  {step.outcome}
                  {step.reason ? ` — ${step.reason}` : ""}
                </p>
              )}

              <FailureStrip failures={run.structuralFailures} />
            </article>
          );
        })}
      </div>
    </section>
  );
}

function FailureStrip({ failures }: { failures: StructuralFailure[] }) {
  if (failures.length === 0) {
    return <p className="sbs-fail is-ok">No structural rejection — draft tree held.</p>;
  }
  return (
    <ul className="sbs-fail-list">
      {failures.map((f) => (
        <li key={`${f.tokenIndex}-${f.kind}`}>
          <strong>{f.label}</strong>
          <span>{f.detail}</span>
        </li>
      ))}
    </ul>
  );
}
