import type { ControlSpec } from "./metaTypes";

export function ControlsPanel(props: {
  controls: ControlSpec[];
  values: Record<string, any>;
  onChange: (key: string, value: any) => void;
}) {
  const { controls, values, onChange } = props;
  if (!controls || controls.length === 0) return null;

  return (
    <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 10, padding: 10 }}>
      <legend style={{ padding: "0 6px" }}>Plot controls</legend>

      <div style={{ display: "grid", gap: 10 }}>
        {controls.map((c) => {
          const label = c.label ?? c.key;
          const v = values?.[c.key] ?? c.default ?? (c.kind === "bool" ? false : "");

          if (c.kind === "bool") {
            return (
              <label key={c.key} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={!!v}
                  onChange={(e) => onChange(c.key, e.target.checked)}
                />
                {label}
              </label>
            );
          }

          if (c.kind === "int" || c.kind === "float") {
            return (
              <label key={c.key}>
                {label}
                <input
                  type="number"
                  value={v}
                  min={c.min}
                  max={c.max}
                  step={c.step ?? (c.kind === "int" ? 1 : 0.1)}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const num = raw === "" ? "" : Number(raw);
                    onChange(c.key, c.kind === "int" && num !== "" ? Math.trunc(num) : num);
                  }}
                  style={{ width: "100%" }}
                />
              </label>
            );
          }

          if (c.kind === "choice") {
            const choices = Array.isArray(c.choices) ? c.choices : [];
            return (
              <label key={c.key}>
                {label}
                <select
                  value={v}
                  onChange={(e) => onChange(c.key, e.target.value)}
                  style={{ width: "100%" }}
                >
                  {choices.map((ch: any, idx: number) => {
                    const opt = typeof ch === "object" ? ch : { label: String(ch), value: ch };
                    return (
                      <option key={idx} value={opt.value}>
                        {opt.label}
                      </option>
                    );
                  })}
                </select>
              </label>
            );
          }

          // text
          return (
            <label key={c.key}>
              {label}
              <input
                value={v ?? ""}
                onChange={(e) => onChange(c.key, e.target.value)}
                style={{ width: "100%" }}
              />
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
