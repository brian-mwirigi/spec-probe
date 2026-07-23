/**
 * Wire format dumped by the Python sampler hook (and the local simulator).
 * Keep this in lockstep with schemas/speculative-trace-v1.json.
 */

export type TraceDraftStrategy =
  | "eagle"
  | "medusa"
  | "ngram"
  | "draft_model"
  | "unknown";

export type TraceOutcome = "accepted" | "rejected" | "unverified" | "bonus";

export type TraceTag =
  | "indent"
  | "syntax"
  | "phrasing"
  | "boundary"
  | "overconfidence"
  | null;

export interface SpeculativeTraceV1 {
  schema_version: "specprobe.trace.v1";
  trace_id: string;
  created_at: string;
  engine: {
    name: string;
    version?: string | null;
    draft_strategy: TraceDraftStrategy;
    target_model?: string | null;
    draft_model?: string | null;
    temperature?: number;
    extra?: Record<string, unknown>;
  };
  request: {
    prompt: string;
    request_id?: string | null;
    domain_hint?: "python" | "prose" | "mixed" | null;
  };
  blocks: TraceBlock[];
  metrics?: {
    wall_time_ms?: number | null;
    acceptance_rate?: number | null;
    tokens_accepted?: number | null;
    tokens_drafted?: number | null;
  };
}

export interface TraceBlock {
  block_index: number;
  draft_len?: number;
  rejected_at?: number | null;
  bonus_token?: {
    token_id?: number | null;
    token: string;
    display?: string;
    target_prob?: number;
  } | null;
  tokens: TraceToken[];
  structural_failures?: TraceFailure[];
}

export interface TraceToken {
  index: number;
  token_id?: number | null;
  token: string;
  display?: string;
  draft_prob: number;
  target_prob: number;
  ratio?: number;
  accept_prob: number;
  roll?: number | null;
  outcome: TraceOutcome;
  tag?: TraceTag;
  reason?: string | null;
}

export interface TraceFailure {
  kind: "indent" | "syntax" | "phrasing" | "boundary" | "overconfidence";
  label: string;
  detail: string;
  token_index: number;
}
