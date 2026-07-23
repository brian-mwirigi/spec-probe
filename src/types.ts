export type DraftStrategy = "eagle" | "medusa" | "ngram";

export type Domain = "python" | "prose" | "mixed";

export type TokenOutcome = "accepted" | "rejected" | "bonus";

export interface TokenStep {
  index: number;
  token: string;
  display: string;
  draftProb: number;
  targetProb: number;
  ratio: number;
  acceptProb: number;
  roll: number;
  outcome: TokenOutcome;
  reason: string | null;
  structuralHint: string | null;
}

export interface ProbeRun {
  id: string;
  scenarioId: string;
  strategy: DraftStrategy;
  temperature: number;
  domain: Domain;
  prompt: string;
  steps: TokenStep[];
  acceptanceRate: number;
  acceptedCount: number;
  rejectedAt: number | null;
  latencyMsEstimate: number;
  structuralFailures: StructuralFailure[];
}

export interface StructuralFailure {
  kind: "indent" | "syntax" | "phrasing" | "boundary" | "overconfidence";
  label: string;
  detail: string;
  tokenIndex: number;
}

export interface Scenario {
  id: string;
  name: string;
  domain: Domain;
  blurb: string;
  prompt: string;
  /** Draft token sequence proposed for one speculative block */
  draftTokens: string[];
  /** Display labels (whitespace-visible) */
  displays: string[];
  /** Base draft logits (pre-temperature) */
  draftLogits: number[];
  /** Target logits aligned to the same tokens */
  targetLogits: number[];
  /** Optional structural tags per token */
  tags?: Array<"indent" | "syntax" | "phrasing" | "boundary" | null>;
}
