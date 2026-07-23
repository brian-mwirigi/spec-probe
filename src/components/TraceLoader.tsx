import { useRef, useState } from "react";
import { FileJson2, Upload } from "lucide-react";

export interface BuiltinTrace {
  id: string;
  label: string;
  path: string;
}

interface Props {
  builtins: BuiltinTrace[];
  activeId: string | null;
  error: string | null;
  onLoadBuiltin: (id: string, path: string) => void;
  onLoadFile: (file: File) => void;
  onClear: () => void;
}

export function TraceLoader({
  builtins,
  activeId,
  error,
  onLoadBuiltin,
  onLoadFile,
  onClear,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <section className="trace-loader" aria-label="Trace input">
      <header className="trace-loader-head">
        <FileJson2 size={18} strokeWidth={1.75} aria-hidden />
        <div>
          <p className="eyebrow">Trace intake · specprobe.trace.v1</p>
          <h2>Load a vLLM dump</h2>
        </div>
      </header>

      <p className="trace-lead">
        Hook intercepts logits + acceptance math, writes JSON, this lane renders it.
        Demo mode still works without a GPU.
      </p>

      <div className="trace-builtins">
        {builtins.map((b) => (
          <button
            key={b.id}
            type="button"
            className={`trace-chip ${activeId === b.id ? "is-active" : ""}`}
            onClick={() => onLoadBuiltin(b.id, b.path)}
          >
            {b.label}
          </button>
        ))}
        {activeId && (
          <button type="button" className="trace-chip ghost" onClick={onClear}>
            Back to simulator
          </button>
        )}
      </div>

      <div
        className={`trace-drop ${dragging ? "is-dragging" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) onLoadFile(file);
        }}
      >
        <Upload size={16} strokeWidth={2} aria-hidden />
        <span>Drop a trace JSON, or</span>
        <button type="button" className="linkish" onClick={() => inputRef.current?.click()}>
          browse
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onLoadFile(file);
            e.target.value = "";
          }}
        />
      </div>

      {error && <p className="trace-error" role="alert">{error}</p>}
    </section>
  );
}
