import type { ProbeRun } from "../types";
import { strategyLabel } from "../lib/scenarios";

interface Props {
  runs: ProbeRun[];
  activeStrategy: string;
  onPick: (strategy: ProbeRun["strategy"]) => void;
}

export function StrategyCompare({ runs, activeStrategy, onPick }: Props) {
  return (
    <div className="strategy-grid">
      {runs.map((run) => {
        const rejected =
          run.rejectedAt !== null
            ? run.steps[run.rejectedAt]?.display
            : "—";
        const active = run.strategy === activeStrategy;
        return (
          <button
            key={run.strategy}
            type="button"
            className={`strategy-card ${active ? "is-active" : ""}`}
            onClick={() => onPick(run.strategy)}
          >
            <span className="strategy-name">{strategyLabel(run.strategy)}</span>
            <span className="strategy-rate">
              {(run.acceptanceRate * 100).toFixed(0)}
              <small>%</small>
            </span>
            <span className="strategy-meta">
              accept {run.acceptedCount}
              {run.rejectedAt !== null ? ` · reject @ ${rejected}` : " · full block"}
            </span>
            <span className="strategy-lat">~{run.latencyMsEstimate} ms est.</span>
          </button>
        );
      })}
    </div>
  );
}
