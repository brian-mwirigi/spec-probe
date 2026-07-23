import type { TokenStep } from "../types";

interface Props {
  steps: TokenStep[];
  selected: number;
  onSelect: (index: number) => void;
}

export function TokenLane({ steps, selected, onSelect }: Props) {
  return (
    <div className="lane" role="list" aria-label="Draft token verification lane">
      {steps.map((step) => {
        const isSelected = step.index === selected;
        const cls = [
          "lane-chip",
          `is-${step.outcome}`,
          isSelected ? "is-selected" : "",
          step.reason?.startsWith("Unverified") ? "is-unverified" : "",
        ]
          .filter(Boolean)
          .join(" ");

        return (
          <button
            key={step.index}
            type="button"
            role="listitem"
            className={cls}
            onClick={() => onSelect(step.index)}
            aria-pressed={isSelected}
            title={step.reason ?? `${step.outcome} · q/p=${Number.isFinite(step.ratio) ? step.ratio.toFixed(2) : "∞"}`}
          >
            <span className="lane-idx">{step.index}</span>
            <span className="lane-tok">{step.display}</span>
            <span className="lane-meta">
              {step.outcome === "bonus"
                ? "bonus"
                : step.reason?.startsWith("Unverified")
                  ? "skip"
                  : `${(step.acceptProb * 100).toFixed(0)}%`}
            </span>
          </button>
        );
      })}
    </div>
  );
}
