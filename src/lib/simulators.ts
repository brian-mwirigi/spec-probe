import {
  acceptanceProbability,
  estimateLatencyMs,
  mulberry32,
  softmax,
} from "./speculative";
import { applyStrategy, SCENARIOS } from "./scenarios";
import type {
  DraftStrategy,
  ProbeRun,
  StructuralFailure,
  TokenStep,
} from "../types";

function reasonForRejection(
  token: string,
  display: string,
  draftProb: number,
  targetProb: number,
  tag: string | null | undefined,
): { reason: string; failure: StructuralFailure | null } {
  const ratio = targetProb / Math.max(draftProb, 1e-12);
  let kind: StructuralFailure["kind"] = "overconfidence";
  let label = "Draft overconfidence";
  let detail = `Draft put ${(draftProb * 100).toFixed(1)}% on ${JSON.stringify(display)} while target only assigned ${(targetProb * 100).toFixed(1)}% (q/p = ${ratio.toFixed(2)}).`;

  if (tag === "indent") {
    kind = "indent";
    label = "Indentation divergence";
    detail = `Whitespace mismatch: draft favored ${JSON.stringify(display)} but the target distribution mass is elsewhere — classic Python indent failure mode.`;
  } else if (tag === "syntax") {
    kind = "syntax";
    label = "Syntax token mismatch";
    detail = `Draft locked onto syntactic glue ${JSON.stringify(display)} too early; target still assigns higher mass to the incomplete structure.`;
  } else if (tag === "phrasing") {
    kind = "phrasing";
    label = "Phrasing divergence";
    detail = `Lexical continuation disagreed: draft ${JSON.stringify(token)} is plausible to the small model but not the mode of the target.`;
  } else if (tag === "boundary") {
    kind = "boundary";
    label = "Domain boundary failure";
    detail = `Draft crossed a prose/code boundary with the wrong register — acceptance collapses until the target resamples.`;
  }

  return {
    reason: `${label}: q/p=${ratio.toFixed(3)} (roll rejected).`,
    failure: { kind, label, detail, tokenIndex: -1 },
  };
}

export interface RunOptions {
  scenarioId: string;
  strategy: DraftStrategy;
  temperature: number;
  seed?: number;
}

export function runProbe(options: RunOptions): ProbeRun {
  const scenario = SCENARIOS.find((s) => s.id === options.scenarioId) ?? SCENARIOS[0];
  const { draftLogits, targetLogits } = applyStrategy(scenario, options.strategy);
  const temperature = options.temperature;
  const rand = mulberry32(options.seed ?? (Date.now() ^ scenario.id.length * 997));

  // Build a tiny competing vocab around each position so softmax is meaningful:
  // position i: [draftToken logit, distractorA, distractorB]
  const steps: TokenStep[] = [];
  const failures: StructuralFailure[] = [];
  let rejectedAt: number | null = null;

  for (let i = 0; i < scenario.draftTokens.length; i++) {
    const dLogit = draftLogits[i];
    const tLogit = targetLogits[i];
    // Distractors sit below the chosen token; gap encodes “peakiness”
    const draftDistractors = [dLogit - 1.4, dLogit - 2.1];
    const targetDistractors = [tLogit - 0.9, tLogit - 1.6];

    const p = softmax([dLogit, ...draftDistractors], temperature)[0];
    const q = softmax([tLogit, ...targetDistractors], temperature)[0];
    const acceptProb = acceptanceProbability(p, q);
    const roll = rand();
    const ratio = q / Math.max(p, 1e-12);
    const tag = scenario.tags?.[i] ?? null;

    const accepted = rejectedAt === null && roll <= acceptProb;
    let outcome: TokenStep["outcome"] = accepted ? "accepted" : "rejected";
    let reason: string | null = null;
    let structuralHint: string | null = null;

    if (rejectedAt !== null) {
      // Tokens after first rejection were never verified in this block
      outcome = "rejected";
      reason = "Unverified — block aborted after earlier rejection.";
    } else if (!accepted) {
      rejectedAt = i;
      const explained = reasonForRejection(
        scenario.draftTokens[i],
        scenario.displays[i],
        p,
        q,
        tag,
      );
      reason = explained.reason;
      structuralHint = explained.failure?.label ?? null;
      if (explained.failure) {
        failures.push({ ...explained.failure, tokenIndex: i });
      }
    } else if (ratio < 0.85 && tag) {
      structuralHint = `Thin accept (${tag}) — q/p=${ratio.toFixed(2)}`;
    }

    steps.push({
      index: i,
      token: scenario.draftTokens[i],
      display: scenario.displays[i],
      draftProb: p,
      targetProb: q,
      ratio,
      acceptProb,
      roll,
      outcome: rejectedAt !== null && i === rejectedAt ? "rejected" : outcome,
      reason,
      structuralHint,
    });

    // Mark subsequent as unverified more clearly
    if (rejectedAt !== null && i > rejectedAt) {
      steps[i].outcome = "rejected";
      steps[i].reason = "Unverified — block aborted after earlier rejection.";
      steps[i].structuralHint = null;
    }

    if (rejectedAt !== null) {
      // Still record remaining proposed tokens for the lane, but stop accepting
      continue;
    }
  }

  // If entire draft accepted, bonus token from target (symbolic)
  if (rejectedAt === null) {
    const bonusP = 0.42 + rand() * 0.2;
    steps.push({
      index: steps.length,
      token: "⟨bonus⟩",
      display: "⟨+1 target⟩",
      draftProb: 0,
      targetProb: bonusP,
      ratio: Infinity,
      acceptProb: 1,
      roll: 0,
      outcome: "bonus",
      reason: "Full draft accepted — target samples one extra token.",
      structuralHint: null,
    });
  }

  const verified = steps.filter((s) => s.outcome !== "bonus" && s.reason !== "Unverified — block aborted after earlier rejection.");
  const acceptedCount = steps.filter((s) => s.outcome === "accepted").length;
  const acceptanceRate =
    verified.length === 0 ? 0 : acceptedCount / Math.max(1, (rejectedAt === null ? scenario.draftTokens.length : rejectedAt + 1));

  return {
    id: `run_${Date.now().toString(36)}`,
    scenarioId: scenario.id,
    strategy: options.strategy,
    temperature,
    domain: scenario.domain,
    prompt: scenario.prompt,
    steps,
    acceptanceRate,
    acceptedCount,
    rejectedAt,
    latencyMsEstimate: estimateLatencyMs(
      scenario.draftTokens.length,
      acceptedCount,
      rejectedAt !== null,
    ),
    structuralFailures: failures,
  };
}

export function compareStrategies(
  scenarioId: string,
  temperature: number,
  seed = 42,
): ProbeRun[] {
  const strategies: DraftStrategy[] = ["eagle", "medusa", "ngram"];
  return strategies.map((strategy, i) =>
    runProbe({ scenarioId, strategy, temperature, seed: seed + i * 17 }),
  );
}
