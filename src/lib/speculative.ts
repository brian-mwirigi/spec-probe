/** Softmax with temperature; returns probabilities for the provided logits. */
export function softmax(logits: number[], temperature: number): number[] {
  const t = Math.max(0.05, temperature);
  const scaled = logits.map((l) => l / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Speculative decoding acceptance probability for a single draft token
 * (Leviathan & Matias / Chen et al.): min(1, q(x) / p(x)).
 */
export function acceptanceProbability(draftProb: number, targetProb: number): number {
  if (draftProb <= 1e-12) return 1;
  return Math.min(1, targetProb / draftProb);
}

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function estimateLatencyMs(
  draftLen: number,
  acceptedCount: number,
  rejected: boolean,
): number {
  // Rough relative cost model: each accepted draft token saves a serial decode.
  const targetForward = 1;
  const draftCost = 0.15 * draftLen;
  const verifyCost = targetForward;
  const recovery = rejected ? 1.1 : 0.2;
  const serialBaseline = (acceptedCount + (rejected ? 1 : 1)) * 12;
  const speculative = (draftCost + verifyCost + recovery) * 12;
  return Math.round(Math.max(8, Math.min(serialBaseline, speculative + (draftLen - acceptedCount) * 3)));
}
