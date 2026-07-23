import type { StructuralFailure } from "../types";

interface Props {
  failures: StructuralFailure[];
}

export function FailureCallouts({ failures }: Props) {
  if (failures.length === 0) {
    return (
      <div className="fail-panel is-clear">
        <p className="eyebrow">Structural failures</p>
        <h3>No structural rejection this block</h3>
        <p className="fail-detail">
          Draft and target stayed distributionally aligned through verification.
        </p>
      </div>
    );
  }

  return (
    <div className="fail-panel">
      <p className="eyebrow">Structural failures</p>
      <ul className="fail-list">
        {failures.map((f) => (
          <li key={`${f.tokenIndex}-${f.kind}`}>
            <span className={`fail-kind kind-${f.kind}`}>{f.label}</span>
            <p>{f.detail}</p>
            <span className="fail-at">token index {f.tokenIndex}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
