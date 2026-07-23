import type { DraftStrategy, ProbeRun, StructuralFailure, TokenStep } from "../types";
import type { SpeculativeTraceV1, TraceBlock, TraceDraftStrategy } from "../types/trace";

const SCHEMA = "specprobe.trace.v1";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function mapStrategy(s: TraceDraftStrategy | string): DraftStrategy {
  if (s === "eagle" || s === "medusa" || s === "ngram") return s;
  return "medusa";
}

function visibleDisplay(token: string, display?: string): string {
  if (display != null && display.length > 0) return display;
  return token
    .replace(/ /g, "␠")
    .replace(/\t/g, "⇥")
    .replace(/\n/g, "↵");
}

function inferTag(token: string): StructuralFailure["kind"] | null {
  if (/^[ \t]+$/.test(token) || token === "\n" || token === "▁" || /^▁+$/.test(token)) {
    return "indent";
  }
  if (/^[\(\)\{\}\[\]:;,.]$/.test(token)) return "syntax";
  if (/^(```|def|class|import|from|return)$/.test(token)) return "boundary";
  return null;
}

function failureFromToken(
  index: number,
  display: string,
  draftProb: number,
  targetProb: number,
  tag: StructuralFailure["kind"] | null,
): StructuralFailure {
  const ratio = targetProb / Math.max(draftProb, 1e-12);
  if (tag === "indent") {
    return {
      kind: "indent",
      label: "Indentation divergence",
      detail: `Whitespace mismatch at ${JSON.stringify(display)} (q/p=${ratio.toFixed(2)}).`,
      tokenIndex: index,
    };
  }
  if (tag === "boundary") {
    return {
      kind: "boundary",
      label: "Domain boundary failure",
      detail: `Register switch near ${JSON.stringify(display)} — draft and target diverged.`,
      tokenIndex: index,
    };
  }
  if (tag === "syntax") {
    return {
      kind: "syntax",
      label: "Syntax token mismatch",
      detail: `Syntactic glue ${JSON.stringify(display)} rejected (q/p=${ratio.toFixed(2)}).`,
      tokenIndex: index,
    };
  }
  return {
    kind: "overconfidence",
    label: "Draft overconfidence",
    detail: `Draft p=${(draftProb * 100).toFixed(1)}% vs target q=${(targetProb * 100).toFixed(1)}% on ${JSON.stringify(display)}.`,
    tokenIndex: index,
  };
}

function blockToRun(trace: SpeculativeTraceV1, block: TraceBlock): ProbeRun {
  const steps: TokenStep[] = [];
  const failures: StructuralFailure[] = [...(block.structural_failures ?? [])].map((f) => ({
    kind: f.kind,
    label: f.label,
    detail: f.detail,
    tokenIndex: f.token_index,
  }));

  let rejectedAt = block.rejected_at ?? null;

  for (const t of block.tokens) {
    const display = visibleDisplay(t.token, t.display);
    const ratio = t.ratio ?? t.target_prob / Math.max(t.draft_prob, 1e-12);
    const outcome =
      t.outcome === "unverified"
        ? "rejected"
        : t.outcome === "bonus"
          ? "bonus"
          : t.outcome;

    let reason = t.reason ?? null;
    if (t.outcome === "unverified" && !reason) {
      reason = "Unverified — block aborted after earlier rejection.";
    }

    const tag =
      (t.tag as StructuralFailure["kind"] | null | undefined) ??
      (t.outcome === "rejected" ? inferTag(t.token) : null);

    if (t.outcome === "rejected" && rejectedAt === null) {
      rejectedAt = t.index;
    }

    if (
      t.outcome === "rejected" &&
      !failures.some((f) => f.tokenIndex === t.index) &&
      reason !== "Unverified — block aborted after earlier rejection."
    ) {
      failures.push(
        failureFromToken(t.index, display, t.draft_prob, t.target_prob, tag),
      );
      if (!reason) {
        reason = `${failures[failures.length - 1].label}: q/p=${ratio.toFixed(3)} (roll rejected).`;
      }
    }

    steps.push({
      index: t.index,
      token: t.token,
      display,
      draftProb: t.draft_prob,
      targetProb: t.target_prob,
      ratio,
      acceptProb: t.accept_prob,
      roll: t.roll ?? 0,
      outcome,
      reason,
      structuralHint: tag,
    });
  }

  if (block.bonus_token && !steps.some((s) => s.outcome === "bonus")) {
    const b = block.bonus_token;
    steps.push({
      index: steps.length,
      token: b.token,
      display: b.display ?? "⟨+1 target⟩",
      draftProb: 0,
      targetProb: b.target_prob ?? 0,
      ratio: Infinity,
      acceptProb: 1,
      roll: 0,
      outcome: "bonus",
      reason: "Full draft accepted — target samples one extra token.",
      structuralHint: null,
    });
  }

  const acceptedCount = steps.filter((s) => s.outcome === "accepted").length;
  const drafted = block.draft_len ?? block.tokens.filter((t) => t.outcome !== "bonus").length;
  const denom = rejectedAt === null ? drafted : rejectedAt + 1;
  const acceptanceRate =
    trace.metrics?.acceptance_rate ??
    (denom <= 0 ? 0 : acceptedCount / denom);

  return {
    id: `${trace.trace_id}#${block.block_index}`,
    scenarioId: "trace",
    strategy: mapStrategy(trace.engine.draft_strategy),
    temperature: trace.engine.temperature ?? 1,
    domain: trace.request.domain_hint ?? "mixed",
    prompt: trace.request.prompt,
    steps,
    acceptanceRate,
    acceptedCount,
    rejectedAt,
    latencyMsEstimate: trace.metrics?.wall_time_ms ?? 0,
    structuralFailures: failures,
  };
}

export function parseTrace(raw: unknown): SpeculativeTraceV1 {
  if (!isRecord(raw)) throw new Error("Trace must be a JSON object.");
  if (raw.schema_version !== SCHEMA) {
    throw new Error(`Unsupported schema_version (expected ${SCHEMA}).`);
  }
  if (!Array.isArray(raw.blocks) || raw.blocks.length === 0) {
    throw new Error("Trace must include at least one block.");
  }
  return raw as unknown as SpeculativeTraceV1;
}

export function traceToProbeRuns(trace: SpeculativeTraceV1): ProbeRun[] {
  return trace.blocks.map((b) => blockToRun(trace, b));
}

export async function loadTraceFile(file: File): Promise<SpeculativeTraceV1> {
  const text = await file.text();
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("File is not valid JSON.");
  }
  return parseTrace(json);
}

export async function fetchTrace(url: string): Promise<SpeculativeTraceV1> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch trace (${res.status}).`);
  return parseTrace(await res.json());
}
