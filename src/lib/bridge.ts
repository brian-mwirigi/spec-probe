import type { SpeculativeTraceV1 } from "../types/trace";
import type { ProbeRun } from "../types";
import { parseTrace, traceToProbeRuns } from "./parseTrace";

export interface SpecEventV1 {
  schema_version: "specprobe.event.v1";
  run_id: string;
  draft_strategy: string;
  prompt?: string;
  draft_tokens: number[];
  /** Optional human-readable displays aligned with draft_tokens */
  displays?: string[];
  tags?: Array<"indent" | "syntax" | "phrasing" | "boundary" | "overconfidence" | null>;
  p_draft: number[];
  p_target: number[];
  accept_prob?: number[];
  rolls?: Array<number | null> | null;
  acceptance_mask: boolean[];
  rejected_at?: number | null;
  accepted_tokens?: number[];
  recovered_token?: number | null;
  bonus_token?: number | null;
  temperature?: number | null;
  draft_probs_available?: boolean;
  step?: number;
  req_index?: number;
  request_id?: string;
  ts?: string;
  domain_hint?: "python" | "prose" | "mixed" | null;
}

export interface StrategyBundle {
  schema_version: string;
  strategies: Record<string, SpeculativeTraceV1[]>;
  count: number;
}

const BRIDGE =
  (import.meta.env.VITE_SPECPROBE_BRIDGE as string | undefined) ||
  "http://127.0.0.1:8787";

function displayId(id: number): string {
  return `id:${id}`;
}

/** Lift a raw rejection-sampler JSONL event into ProbeRun without the Python bridge. */
export function eventToProbeRun(event: SpecEventV1): ProbeRun {
  const steps: ProbeRun["steps"] = event.draft_tokens.map((tok, i) => {
    const draftProb = event.p_draft[i] ?? 0;
    const targetProb = event.p_target[i] ?? 0;
    const ratio = targetProb / Math.max(draftProb, 1e-12);
    const acceptProb = event.accept_prob?.[i] ?? Math.min(1, ratio);
    const accepted = event.acceptance_mask[i] ?? false;
    const rejectedAt = event.rejected_at ?? null;
    const display = event.displays?.[i] ?? displayId(tok);
    const tag = event.tags?.[i] ?? null;
    let outcome: ProbeRun["steps"][number]["outcome"] = accepted ? "accepted" : "rejected";
    let reason: string | null = null;
    if (rejectedAt !== null && i > rejectedAt) {
      outcome = "rejected";
      reason = "Unverified — block aborted after earlier rejection.";
    } else if (!accepted) {
      reason =
        tag === "indent"
          ? `Indentation divergence: q/p=${ratio.toFixed(3)}`
          : `Rejected: q/p=${ratio.toFixed(3)}`;
    }
    return {
      index: i,
      token: display,
      display,
      draftProb,
      targetProb,
      ratio,
      acceptProb,
      roll: event.rolls?.[i] ?? 0,
      outcome,
      reason,
      structuralHint: tag,
    };
  });

  if (event.bonus_token != null && event.rejected_at == null) {
    steps.push({
      index: steps.length,
      token: displayId(event.bonus_token),
      display: "⟨+1 target⟩",
      draftProb: 0,
      targetProb: 0,
      ratio: Infinity,
      acceptProb: 1,
      roll: 0,
      outcome: "bonus",
      reason: "Full draft accepted — target samples one extra token.",
      structuralHint: null,
    });
  }

  const acceptedCount = steps.filter((s) => s.outcome === "accepted").length;
  const verified = steps.filter(
    (s) =>
      s.outcome === "accepted" ||
      (s.outcome === "rejected" && !s.reason?.startsWith("Unverified")),
  ).length;

  const strategy =
    event.draft_strategy === "eagle" ||
    event.draft_strategy === "medusa" ||
    event.draft_strategy === "ngram"
      ? event.draft_strategy
      : event.draft_strategy.startsWith("eagle")
        ? "eagle"
        : "medusa";

  const rejectTag = event.rejected_at != null ? event.tags?.[event.rejected_at] : null;

  return {
    id: `${event.run_id}_${event.step ?? 0}`,
    scenarioId: "live",
    strategy,
    temperature: event.temperature ?? 1,
    domain: event.domain_hint ?? "mixed",
    prompt: event.prompt ?? "",
    steps,
    acceptanceRate: verified ? acceptedCount / verified : 0,
    acceptedCount,
    rejectedAt: event.rejected_at ?? null,
    latencyMsEstimate: 0,
    structuralFailures:
      event.rejected_at != null
        ? [
            {
              kind: rejectTag === "indent" ? "indent" : "overconfidence",
              label:
                rejectTag === "indent"
                  ? "Indentation divergence"
                  : event.draft_strategy === "ngram"
                    ? "n-gram lookahead collapse"
                    : "Draft overconfidence",
              detail:
                rejectTag === "indent"
                  ? `Whitespace mismatch at draft index ${event.rejected_at}; recovered_token=${event.recovered_token ?? "n/a"}.`
                  : `First rejection at index ${event.rejected_at}; recovered_token=${event.recovered_token ?? "n/a"}.`,
              tokenIndex: event.rejected_at,
            },
          ]
        : [],
  };
}

export async function fetchBundle(bridge = BRIDGE): Promise<StrategyBundle> {
  const res = await fetch(`${bridge}/bundle`);
  if (!res.ok) throw new Error(`Bridge /bundle failed (${res.status})`);
  return res.json();
}

export async function pollEvents(
  offset: number,
  bridge = BRIDGE,
): Promise<{ events: SpecEventV1[]; offset: number }> {
  const res = await fetch(`${bridge}/events?offset=${offset}`);
  if (!res.ok) throw new Error(`Bridge /events failed (${res.status})`);
  return res.json();
}

export async function loadLocalJsonl(url: string): Promise<ProbeRun[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  const text = await res.text();
  const runs: ProbeRun[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const ev = JSON.parse(trimmed) as SpecEventV1;
    runs.push(eventToProbeRun(ev));
  }
  return runs;
}

export function bundleToRuns(bundle: StrategyBundle): Record<string, ProbeRun[]> {
  const out: Record<string, ProbeRun[]> = {};
  for (const [strategy, traces] of Object.entries(bundle.strategies)) {
    out[strategy] = traces.flatMap((t) => {
      try {
        return traceToProbeRuns(parseTrace(t));
      } catch {
        return [];
      }
    });
  }
  return out;
}
