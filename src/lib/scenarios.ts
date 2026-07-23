import type { Domain, DraftStrategy, Scenario } from "../types";

export const SCENARIOS: Scenario[] = [
  {
    id: "py-indent",
    name: "Python indentation drift",
    domain: "python",
    blurb: "Draft overcommits to a wrong indent level; target mass sits on spaces/newlines elsewhere.",
    prompt: "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) +",
    draftTokens: ["▁", "▁", "fib", "(", "n", "-", "2", ")", "\n", "▁▁"],
    displays: ["␠", "␠", "fib", "(", "n", "-", "2", ")", "↵", "␠␠"],
    draftLogits: [4.2, 3.9, 5.1, 4.8, 4.6, 3.2, 4.0, 4.7, 2.8, 4.5],
    targetLogits: [1.1, 0.9, 5.4, 5.0, 4.9, 3.5, 4.2, 5.1, 4.8, 0.4],
    tags: ["indent", "indent", null, "syntax", null, null, null, "syntax", "boundary", "indent"],
  },
  {
    id: "prose-flow",
    name: "Natural language phrasing",
    domain: "prose",
    blurb: "Draft and target stay aligned on ordinary English continuations — high acceptance.",
    prompt: "Speculative decoding speeds up generation by letting a small model propose tokens that a",
    draftTokens: ["▁larger", "▁model", "▁can", "▁verify", "▁in", "▁parallel", "."],
    displays: [" larger", " model", " can", " verify", " in", " parallel", "."],
    draftLogits: [4.6, 4.9, 4.4, 4.7, 4.1, 4.8, 3.9],
    targetLogits: [4.5, 5.0, 4.3, 4.9, 4.0, 4.7, 4.1],
    tags: ["phrasing", "phrasing", "phrasing", "phrasing", "phrasing", "phrasing", null],
  },
  {
    id: "mixed-boundary",
    name: "Prose → code boundary",
    domain: "mixed",
    blurb: "Acceptance collapses at the language switch when the draft keeps writing prose into a code fence.",
    prompt: "Here is a minimal example:\n\n```python\n",
    draftTokens: ["The", "▁following", "▁function", "▁computes", "\n", "def", "▁add", "(", "a", ","],
    displays: ["The", " following", " function", " computes", "↵", "def", " add", "(", "a", ","],
    draftLogits: [5.2, 4.8, 4.6, 4.4, 2.0, 3.1, 3.0, 2.8, 2.5, 2.4],
    targetLogits: [0.6, 0.5, 0.4, 0.3, 3.8, 5.5, 4.9, 4.7, 4.6, 4.4],
    tags: ["boundary", "phrasing", "phrasing", "phrasing", "boundary", "syntax", "syntax", "syntax", null, null],
  },
  {
    id: "py-syntax",
    name: "Closing punctuation overconfidence",
    domain: "python",
    blurb: "N-gram style draft piles probability on `:` / `)` early; target still wants the argument list.",
    prompt: "for item in items",
    draftTokens: [":", "\n", "▁▁▁▁", "print", "(", "item", ")", "\n"],
    displays: [":", "↵", "␠␠␠␠", "print", "(", "item", ")", "↵"],
    draftLogits: [5.8, 4.2, 4.0, 3.5, 3.2, 3.0, 2.8, 2.5],
    targetLogits: [2.1, 1.0, 0.8, 1.2, 1.0, 0.9, 0.7, 0.5],
    tags: ["syntax", "indent", "indent", "syntax", "syntax", null, "syntax", "boundary"],
  },
];

/** Strategy modulates how closely draft logits track the target. */
export function applyStrategy(
  scenario: Scenario,
  strategy: DraftStrategy,
): { draftLogits: number[]; targetLogits: number[] } {
  const target = [...scenario.targetLogits];
  const draft = scenario.draftLogits.map((d, i) => {
    const t = target[i];
    switch (strategy) {
      case "eagle":
        // High-fidelity draft: pull toward target
        return d * 0.35 + t * 0.65;
      case "medusa":
        // Mid fidelity
        return d * 0.7 + t * 0.3;
      case "ngram":
        // Aggressive / peaky draft — amplify overconfidence away from target
        return d * 1.25 - t * 0.15;
      default:
        return d;
    }
  });
  return { draftLogits: draft, targetLogits: target };
}

export function domainLabel(domain: Domain): string {
  switch (domain) {
    case "python":
      return "Python / code";
    case "prose":
      return "Natural language";
    case "mixed":
      return "Mixed boundary";
  }
}

export function strategyLabel(strategy: DraftStrategy): string {
  switch (strategy) {
    case "eagle":
      return "EAGLE";
    case "medusa":
      return "Medusa";
    case "ngram":
      return "n-gram lookahead";
  }
}
