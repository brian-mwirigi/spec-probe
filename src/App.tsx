import { useEffect, useMemo, useState } from "react";
import { Activity, Play, Radar, Radio } from "lucide-react";
import { TokenLane } from "./components/TokenLane";
import { ProbCompare } from "./components/ProbCompare";
import { AcceptanceCurve } from "./components/AcceptanceCurve";
import { StrategyCompare } from "./components/StrategyCompare";
import { FailureCallouts } from "./components/FailureCallouts";
import { TraceLoader, type BuiltinTrace } from "./components/TraceLoader";
import { SideBySide } from "./components/SideBySide";
import { SCENARIOS, domainLabel, strategyLabel } from "./lib/scenarios";
import { compareStrategies, runProbe } from "./lib/simulators";
import { fetchTrace, loadTraceFile, traceToProbeRuns } from "./lib/parseTrace";
import { eventToProbeRun, loadLocalJsonl, pollEvents, type SpecEventV1 } from "./lib/bridge";
import type { SpeculativeTraceV1 } from "./types/trace";
import type { DraftStrategy, ProbeRun } from "./types";

const BUILTIN_TRACES: BuiltinTrace[] = [
  {
    id: "py-indent",
    label: "py-indent · medusa",
    path: "/traces/py-indent.medusa.json",
  },
  {
    id: "prose-flow",
    label: "prose-flow · eagle",
    path: "/traces/prose-flow.eagle.json",
  },
  {
    id: "mixed-boundary",
    label: "mixed-boundary · ngram",
    path: "/traces/mixed-boundary.ngram.json",
  },
  {
    id: "live-jsonl",
    label: "indent drift · ngram vs eagle",
    path: "/traces/live.jsonl",
  },
];

const AUTO_COMPARE = BUILTIN_TRACES.find((t) => t.id === "live-jsonl")!;

export default function App() {
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const [strategy, setStrategy] = useState<DraftStrategy>("medusa");
  const [temperature, setTemperature] = useState(0.8);
  const [seed, setSeed] = useState(42);
  const [selected, setSelected] = useState(0);

  const [trace, setTrace] = useState<SpeculativeTraceV1 | null>(null);
  const [traceSourceId, setTraceSourceId] = useState<string | null>(null);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [blockIndex, setBlockIndex] = useState(0);

  const [compareRuns, setCompareRuns] = useState<ProbeRun[]>([]);
  const [sbsSelected, setSbsSelected] = useState<Record<string, number>>({});
  const [liveEvents, setLiveEvents] = useState<SpecEventV1[]>([]);
  const [liveOffset, setLiveOffset] = useState(0);
  const [liveOn, setLiveOn] = useState(false);
  const [liveStatus, setLiveStatus] = useState("bridge idle");

  // Boot straight into the indent-drift side-by-side (EAGLE crushes n-gram).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const runs = await loadLocalJsonl(AUTO_COMPARE.path);
        if (cancelled) return;
        setCompareRuns(runs);
        setTrace(null);
        setTraceSourceId(AUTO_COMPARE.id);
        setTraceError(null);
        setSelected(0);
      } catch (err) {
        if (!cancelled) {
          setTraceError(err instanceof Error ? err.message : "Failed to load compare.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const scenario = SCENARIOS.find((s) => s.id === scenarioId) ?? SCENARIOS[0];
  const liveMode = trace !== null;

  const simRun = useMemo(
    () => runProbe({ scenarioId, strategy, temperature, seed }),
    [scenarioId, strategy, temperature, seed],
  );

  const comparisons = useMemo(
    () => compareStrategies(scenarioId, temperature, seed),
    [scenarioId, temperature, seed],
  );

  const traceRuns: ProbeRun[] = useMemo(
    () => (trace ? traceToProbeRuns(trace) : []),
    [trace],
  );

  const run = liveMode
    ? traceRuns[Math.min(blockIndex, Math.max(0, traceRuns.length - 1))] ?? simRun
    : simRun;

  const selectedStep = run.steps[Math.min(selected, run.steps.length - 1)] ?? null;

  const sbsLanes = useMemo(() => {
    const source = compareRuns.length
      ? compareRuns
      : liveEvents.map(eventToProbeRun);
    const byStrategy = new Map<string, ProbeRun>();
    for (const r of source) {
      if (!byStrategy.has(r.strategy)) byStrategy.set(r.strategy, r);
    }
    return [...byStrategy.entries()].map(([strategyName, probeRun]) => ({
      strategy: strategyName,
      run: probeRun,
    }));
  }, [compareRuns, liveEvents]);

  useEffect(() => {
    if (!liveOn) return;
    let cancelled = false;
    let offset = liveOffset;

    async function tick() {
      try {
        const { events, offset: next } = await pollEvents(offset, "/bridge");
        if (cancelled) return;
        if (events.length) {
          setLiveEvents((prev) => [...prev, ...events]);
          setLiveStatus(`live · +${events.length} · offset ${next}`);
        } else {
          setLiveStatus(`live · watching · offset ${offset}`);
        }
        offset = next;
        setLiveOffset(next);
      } catch {
        if (!cancelled) setLiveStatus("bridge unreachable — is :8787 up?");
      }
    }

    tick();
    const id = window.setInterval(tick, 750);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [liveOn]); // eslint-disable-line react-hooks/exhaustive-deps

  function probe() {
    setSeed((s) => s + 1);
    setSelected(0);
  }

  async function applyTrace(next: SpeculativeTraceV1, sourceId: string | null) {
    setTrace(next);
    setTraceSourceId(sourceId);
    setTraceError(null);
    setBlockIndex(0);
    setSelected(0);
    setCompareRuns([]);
  }

  async function onLoadBuiltin(id: string, path: string) {
    try {
      if (path.endsWith(".jsonl")) {
        const runs = await loadLocalJsonl(path);
        setCompareRuns(runs);
        setTrace(null);
        setTraceSourceId(id);
        setTraceError(null);
        setSelected(0);
        if (runs[0]) {
          setStrategy(runs[0].strategy);
        }
        return;
      }
      const next = await fetchTrace(path);
      await applyTrace(next, id);
    } catch (err) {
      setTraceError(err instanceof Error ? err.message : "Failed to load trace.");
    }
  }

  async function onLoadFile(file: File) {
    try {
      if (file.name.endsWith(".jsonl")) {
        const text = await file.text();
        const runs: ProbeRun[] = [];
        for (const line of text.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          runs.push(eventToProbeRun(JSON.parse(trimmed) as SpecEventV1));
        }
        setCompareRuns(runs);
        setTrace(null);
        setTraceSourceId(`file:${file.name}`);
        setTraceError(null);
        return;
      }
      const next = await loadTraceFile(file);
      await applyTrace(next, `file:${file.name}`);
    } catch (err) {
      setTraceError(err instanceof Error ? err.message : "Failed to parse trace.");
    }
  }

  function onClearTrace() {
    setTrace(null);
    setTraceSourceId(null);
    setTraceError(null);
    setBlockIndex(0);
    setSelected(0);
    setCompareRuns([]);
  }

  return (
    <div className="shell">
      <div className="atmosphere" aria-hidden />

      <header className="hero">
        <div className="brand-lockup">
          <Radar className="brand-mark" strokeWidth={1.5} aria-hidden />
          <h1 className="brand">spec-probe</h1>
        </div>
        <p className="tagline">
          Gut vLLM&apos;s rejection sampler — token-level p_draft / p_target, not the macro acceptance rate.
        </p>
        <div className="hero-cta">
          {!liveMode && compareRuns.length === 0 && (
            <button type="button" className="btn primary" onClick={probe}>
              <Play size={16} strokeWidth={2} aria-hidden />
              Run probe
            </button>
          )}
          <button
            type="button"
            className="btn primary"
            onClick={() => setLiveOn((v) => !v)}
          >
            <Radio size={16} strokeWidth={2} aria-hidden />
            {liveOn ? "Stop bridge" : "Stream JSONL"}
          </button>
          <span className="hero-note">
            Patch <code>vllm.v1.sample.rejection_sampler.rejection_sample</code> → JSONL → this lane.
          </span>
        </div>
        <div className="live-bar">
          <span className={`status ${liveOn ? "is-hot" : ""}`}>{liveStatus}</span>
        </div>
      </header>

      <TraceLoader
        builtins={BUILTIN_TRACES}
        activeId={traceSourceId}
        error={traceError}
        onLoadBuiltin={onLoadBuiltin}
        onLoadFile={onLoadFile}
        onClear={onClearTrace}
      />

      {sbsLanes.length > 1 && (
        <SideBySide
          lanes={sbsLanes}
          selected={sbsSelected}
          prompt={sbsLanes[0]?.run.prompt}
          onSelect={(strat, index) =>
            setSbsSelected((prev) => ({ ...prev, [strat]: index }))
          }
        />
      )}

      {!liveMode && compareRuns.length === 0 && (
        <section className="controls" aria-label="Probe controls">
          <label className="field">
            <span>Scenario</span>
            <select
              value={scenarioId}
              onChange={(e) => {
                setScenarioId(e.target.value);
                setSelected(0);
              }}
            >
              {SCENARIOS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Draft strategy</span>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as DraftStrategy)}
            >
              <option value="eagle">EAGLE</option>
              <option value="medusa">Medusa</option>
              <option value="ngram">n-gram lookahead</option>
            </select>
          </label>

          <label className="field field-temp">
            <span>Temperature {temperature.toFixed(2)}</span>
            <input
              type="range"
              min={0.2}
              max={1.4}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
          </label>
        </section>
      )}

      {liveMode && traceRuns.length > 1 && (
        <section className="controls" aria-label="Block picker">
          <label className="field">
            <span>Speculative block</span>
            <select
              value={blockIndex}
              onChange={(e) => {
                setBlockIndex(Number(e.target.value));
                setSelected(0);
              }}
            >
              {traceRuns.map((r, i) => (
                <option key={r.id} value={i}>
                  Block {i} · accept {(r.acceptanceRate * 100).toFixed(0)}%
                </option>
              ))}
            </select>
          </label>
        </section>
      )}

      <section className="prompt-strip">
        <div>
          <p className="eyebrow">
            {liveMode
              ? `Trace prompt · ${domainLabel(run.domain)}`
              : compareRuns[0]
                ? `Sweep prompt · ${domainLabel(compareRuns[0].domain)}`
                : `Prompt prefix · ${domainLabel(scenario.domain)}`}
          </p>
          <pre className="prompt">
            {compareRuns[0]?.prompt ?? run.prompt}
          </pre>
          <p className="scenario-blurb">
            {liveMode
              ? `Schema ${trace.schema_version} · dumped by ${trace.engine.name}${trace.engine.version ? ` ${trace.engine.version}` : ""}.`
              : compareRuns.length
                ? "JSONL events from the rejection-sampler hook (draft_tokens, p_draft, p_target, acceptance_mask, bonus)."
                : scenario.blurb}
          </p>
        </div>
        <dl className="run-kpis">
          <div>
            <dt>Acceptance</dt>
            <dd className={run.acceptanceRate < 0.5 ? "warn" : ""}>
              {(run.acceptanceRate * 100).toFixed(0)}%
            </dd>
          </div>
          <div>
            <dt>Strategy</dt>
            <dd>{strategyLabel(run.strategy)}</dd>
          </div>
          <div>
            <dt>{liveMode ? "Wall time" : "Est. latency"}</dt>
            <dd>~{Math.round(run.latencyMsEstimate)} ms</dd>
          </div>
          <div>
            <dt>Reject @</dt>
            <dd>{run.rejectedAt === null ? "none" : run.rejectedAt}</dd>
          </div>
        </dl>
      </section>

      <section className="probe-stage">
        <div className="stage-head">
          <Activity size={18} strokeWidth={1.75} aria-hidden />
          <h2>Verification lane</h2>
          <p>Click a token to inspect draft p(x) vs target q(x).</p>
        </div>
        <TokenLane
          steps={(compareRuns[0] ?? run).steps}
          selected={selectedStep?.index ?? 0}
          onSelect={setSelected}
        />
        <div className="legend">
          <span className="leg is-accepted">accepted</span>
          <span className="leg is-rejected">rejected</span>
          <span className="leg is-unverified">unverified</span>
          <span className="leg is-bonus">bonus</span>
        </div>
      </section>

      <div className="detail-grid">
        <ProbCompare
          step={
            compareRuns[0]
              ? compareRuns[0].steps[Math.min(selected, compareRuns[0].steps.length - 1)]
              : selectedStep
          }
        />
        <AcceptanceCurve run={compareRuns[0] ?? run} />
        <FailureCallouts failures={(compareRuns[0] ?? run).structuralFailures} />
      </div>

      {!liveMode && compareRuns.length === 0 && (
        <section className="compare-section">
          <header>
            <p className="eyebrow">Strategy sweep · same scenario & temperature</p>
            <h2>EAGLE vs Medusa vs n-gram</h2>
            <p className="compare-lead">
              Simulator preview — replace with hooked JSONL from <code>spec-probe-sweep</code> for artifact-grade lanes.
            </p>
          </header>
          <StrategyCompare
            runs={comparisons}
            activeStrategy={strategy}
            onPick={(s) => {
              setStrategy(s);
              setSelected(0);
            }}
          />
        </section>
      )}

      <footer className="foot">
        <p>
          Injection point: <code>vllm.v1.sample.rejection_sampler.rejection_sample</code>.
          Transport: append-only JSONL. Bridge: FastAPI <code>/events</code> + <code>/ws</code>.
        </p>
      </footer>
    </div>
  );
}
