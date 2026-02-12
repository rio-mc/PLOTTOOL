import { useEffect, useRef, useState } from "react";
import { defaultSeries, defaultSpec } from "./defaultSpec";
import type { PlotSpec, SeriesSpec } from "./types";
import type { FamilyItem, PlotTypeItem, PlotTypeMeta, RenderJsonResponse } from "./metaTypes";
import { ControlsPanel } from "./ControlsPanel";

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "http://127.0.0.1:8000";

const MARKERS: Array<[string, string]> = [
  ["Circle (o)", "o"],
  ["Square (s)", "s"],
  ["Triangle up (^)", "^"],
  ["Triangle down (v)", "v"],
  ["Diamond (D)", "D"],
  ["Plus (+)", "+"],
  ["Cross (x)", "x"],
  ["Point (.)", "."],
  ["None", ""],
];

const LINESTYLES: Array<[string, string]> = [
  ["Solid", "solid"],
  ["Dashed", "dashed"],
  ["Dotted", "dotted"],
  ["Dash-dot", "dashdot"],
];

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return (await res.json()) as T;
}

export default function App() {
  // ---- meta lists ----
  const [families, setFamilies] = useState<FamilyItem[]>([]);
  const [types, setTypes] = useState<PlotTypeItem[]>([]);
  const [fonts, setFonts] = useState<string[]>([]);
  const [plotTypeMeta, setPlotTypeMeta] = useState<PlotTypeMeta | null>(null);

  // ---- ONE source of truth ----
  const [spec, setSpec] = useState<PlotSpec>(() => defaultSpec());

  // 3D view controls (request-level)
  const [viewElev, setViewElev] = useState(30.0);
  const [viewAzim, setViewAzim] = useState(-60.0);
  const [viewRoll, setViewRoll] = useState(0.0);

  // rendering
  const [previewDpi, setPreviewDpi] = useState(160);
  const [imgSrc, setImgSrc] = useState("");
  const [issues, setIssues] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<number | null>(null);

  // ---- Load families + fonts on mount ----
  useEffect(() => {
    (async () => {
      try {
        const fam = await getJson<FamilyItem[]>("/meta/families");
        setFamilies(fam);

        // If current spec family isn't available, snap to first
        if (fam.length && !fam.find((f) => f.value === spec.plot_family)) {
          setSpec((s) => ({ ...s, plot_family: fam[0].value }));
        }

        const fnts = await getJson<string[]>("/meta/fonts");
        setFonts(fnts);

        // If current font not available, snap
        if (fnts.length && !fnts.includes(spec.style.font_family)) {
          setSpec((s) => ({
            ...s,
            style: {
              ...s.style,
              font_family: fnts.includes("Times New Roman") ? "Times New Roman" : fnts[0],
            },
          }));
        }
      } catch (e: any) {
        setError(e?.message ?? String(e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Load plot types whenever family changes ----
  useEffect(() => {
    (async () => {
      try {
        const t = await getJson<PlotTypeItem[]>(
          `/meta/types?family=${encodeURIComponent(spec.plot_family)}`
        );
        setTypes(t);

        // snap to first if current not present
        if (t.length && !t.find((x) => x.value === spec.plot_type)) {
          setSpec((s) => ({ ...s, plot_type: t[0].value }));
        }
      } catch (e: any) {
        setError(e?.message ?? String(e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.plot_family]);

  // ---- Update plotTypeMeta whenever plotType/types changes ----
  useEffect(() => {
    const item = types.find((t) => t.value === spec.plot_type);
    setPlotTypeMeta(item?.meta ?? null);
  }, [spec.plot_type, types]);

  const is3d = !!plotTypeMeta?.requires_z;
  const showX = plotTypeMeta?.requires_x !== false;
  const valuesOnly = !!plotTypeMeta?.y_is_values_only || !showX;

  // ---- Keep series array sized to spec.series_count ----
  useEffect(() => {
    setSpec((s) => {
      const count = clamp(Number(s.series_count || 1), 1, 10);
      const next = [...(s.series ?? [])];
      while (next.length < count) next.push(defaultSeries(next.length));
      while (next.length > count) next.pop();
      return { ...s, series_count: count, series: next };
    });
  }, [spec.series_count]);

  function patchStyle(patch: Partial<PlotSpec["style"]>) {
    setSpec((s) => ({ ...s, style: { ...s.style, ...patch } }));
  }

  function patchSeries(i: number, patch: Partial<SeriesSpec>) {
    setSpec((s) => ({
      ...s,
      series: s.series.map((ser, idx) => (idx === i ? { ...ser, ...patch } : ser)),
    }));
  }

  function patchSeriesInline(i: number, patch: Partial<SeriesSpec["inline"]>) {
    setSpec((s) => ({
      ...s,
      series: s.series.map((ser, idx) =>
        idx === i ? { ...ser, inline: { ...ser.inline, ...patch } } : ser
      ),
    }));
  }

  function patchSeriesStyle(i: number, patch: Partial<SeriesSpec["style"]>) {
    setSpec((s) => ({
      ...s,
      series: s.series.map((ser, idx) =>
        idx === i ? { ...ser, style: { ...ser.style, ...patch } } : ser
      ),
    }));
  }

  async function renderPreview(currentSpec: PlotSpec) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/render_json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec: currentSpec,
          format: "png",
          dpi: previewDpi,
          elev: is3d ? viewElev : undefined,
          azim: is3d ? viewAzim : undefined,
          roll: is3d ? viewRoll : undefined,
        }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
      const data = (await res.json()) as RenderJsonResponse;
      setImgSrc(`data:${data.mime};base64,${data.payload_base64}`);
      setIssues(data.issues ?? []);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setImgSrc("");
      setIssues([]);
    } finally {
      setLoading(false);
    }
  }

  // Debounced like Qt (120ms)
  useEffect(() => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    const snap = spec; // snapshot
    debounceRef.current = window.setTimeout(() => {
      renderPreview(snap);
    }, 140);
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, previewDpi, viewElev, viewAzim, viewRoll, is3d]);

  async function exportFigure(format: "png" | "pdf" | "svg") {
    const dpi = format === "png" ? 300 : 300;
    const res = await fetch(`${API_BASE}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spec,
        format,
        dpi,
        elev: is3d ? viewElev : undefined,
        azim: is3d ? viewAzim : undefined,
        roll: is3d ? viewRoll : undefined,
      }),
    });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `figure.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function exportCode() {
    const res = await fetch(`${API_BASE}/export/code`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "plot.py";
    a.click();
    URL.revokeObjectURL(url);
  }

  function setPreset(p: "reset" | "xy" | "xz" | "yz") {
    if (p === "reset") {
      setViewElev(30);
      setViewAzim(-60);
      setViewRoll(0);
    } else if (p === "xy") {
      setViewElev(90);
      setViewAzim(-90);
      setViewRoll(0);
    } else if (p === "xz") {
      setViewElev(0);
      setViewAzim(-90);
      setViewRoll(0);
    } else if (p === "yz") {
      setViewElev(0);
      setViewAzim(0);
      setViewRoll(0);
    }
  }

  return (
    <div
      style={{
        height: "100vh",
        display: "grid",
        gridTemplateColumns: "380px 1fr 420px",
        gap: 12,
        padding: 12,
      }}
    >
      {/* Left: controls */}
      <div style={{ overflow: "auto", border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
        <h3 style={{ marginTop: 0 }}>Controls</h3>

        <div style={{ display: "grid", gap: 10 }}>
          <label>
            Plot family
            <select
              value={spec.plot_family}
              onChange={(e) => setSpec((s) => ({ ...s, plot_family: e.target.value }))}
              style={{ width: "100%" }}
            >
              {families.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Plot type
            <select
              value={spec.plot_type}
              onChange={(e) => setSpec((s) => ({ ...s, plot_type: e.target.value }))}
              style={{ width: "100%" }}
            >
              {types.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>

          {/* Plot-specific controls (will show once meta.controls non-empty) */}
          <ControlsPanel
            controls={plotTypeMeta?.controls ?? []}
            values={spec.settings.options}
            onChange={(key, value) =>
              setSpec((s) => ({
                ...s,
                settings: { ...s.settings, options: { ...s.settings.options, [key]: value } },
              }))
            }
          />

          <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 10, padding: 10 }}>
            <legend style={{ padding: "0 6px" }}>Style</legend>

            <label>
              Font
              <input
                list="fonts"
                value={spec.style.font_family}
                onChange={(e) => patchStyle({ font_family: e.target.value })}
                style={{ width: "100%" }}
              />
              <datalist id="fonts">
                {fonts.map((f) => (
                  <option key={f} value={f} />
                ))}
              </datalist>
            </label>

            <div style={{ marginTop: 10 }}>
              <div>Title</div>
              <div style={{ display: "flex", gap: 6 }}>
                <input
                  value={spec.style.title}
                  onChange={(e) => patchStyle({ title: e.target.value })}
                  style={{ flex: 1 }}
                />
                <button
                  onClick={() => patchStyle({ title_bold: !spec.style.title_bold })}
                  style={{ fontWeight: "bold", opacity: spec.style.title_bold ? 1 : 0.5 }}
                >
                  B
                </button>
                <button
                  onClick={() => patchStyle({ title_italic: !spec.style.title_italic })}
                  style={{ fontStyle: "italic", opacity: spec.style.title_italic ? 1 : 0.5 }}
                >
                  I
                </button>
                <button
                  onClick={() => patchStyle({ title_underline: !spec.style.title_underline })}
                  style={{
                    textDecoration: "underline",
                    opacity: spec.style.title_underline ? 1 : 0.5,
                  }}
                >
                  U
                </button>
              </div>
            </div>

            <label style={{ marginTop: 10 }}>
              x label
              <input
                value={spec.style.x_label}
                onChange={(e) => patchStyle({ x_label: e.target.value })}
                style={{ width: "100%" }}
              />
            </label>

            <label>
              y label
              <input
                value={spec.style.y_label}
                onChange={(e) => patchStyle({ y_label: e.target.value })}
                style={{ width: "100%" }}
              />
            </label>

            <label>
              x tick label angle
              <input
                type="number"
                value={spec.style.x_tick_label_angle}
                onChange={(e) =>
                  patchStyle({ x_tick_label_angle: clamp(parseInt(e.target.value || "0", 10), -180, 180) })
                }
                style={{ width: "100%" }}
              />
            </label>

            <label>
              y tick label angle
              <input
                type="number"
                value={spec.style.y_tick_label_angle}
                onChange={(e) =>
                  patchStyle({ y_tick_label_angle: clamp(parseInt(e.target.value || "0", 10), -180, 180) })
                }
                style={{ width: "100%" }}
              />
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
              <input
                type="checkbox"
                checked={spec.style.show_grid}
                onChange={(e) => patchStyle({ show_grid: e.target.checked })}
              />
              Grid (major)
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={spec.style.show_minor_ticks}
                onChange={(e) => {
                  const checked = e.target.checked;
                  patchStyle({
                    show_minor_ticks: checked,
                    show_minor_grid: checked ? spec.style.show_minor_grid : false,
                  });
                }}
              />
              Minor ticks
            </label>

            <label
              style={{
                display: "flex",
                gap: 8,
                alignItems: "center",
                opacity: spec.style.show_minor_ticks ? 1 : 0.5,
              }}
            >
              <input
                type="checkbox"
                checked={spec.style.show_minor_grid}
                disabled={!spec.style.show_minor_ticks}
                onChange={(e) => patchStyle({ show_minor_grid: e.target.checked })}
              />
              Grid (minor)
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={spec.style.show_legend}
                onChange={(e) => patchStyle({ show_legend: e.target.checked })}
              />
              Legend
            </label>

            <label>
              Text font size
              <input
                type="number"
                value={spec.style.base_font_size}
                onChange={(e) =>
                  patchStyle({ base_font_size: clamp(parseInt(e.target.value || "11", 10), 7, 36) })
                }
                style={{ width: "100%" }}
              />
            </label>

            <label>
              Title font size
              <input
                type="number"
                value={spec.style.title_font_size}
                onChange={(e) =>
                  patchStyle({ title_font_size: clamp(parseInt(e.target.value || "13", 10), 7, 48) })
                }
                style={{ width: "100%" }}
              />
            </label>

            <label>
              Outlier k
              <input
                type="number"
                step="0.25"
                value={spec.style.outlier_sigma}
                onChange={(e) => patchStyle({ outlier_sigma: parseFloat(e.target.value || "3") })}
                style={{ width: "100%" }}
              />
            </label>

            <label>
              Outlier method
              <select
                value={spec.style.outlier_method}
                onChange={(e) => patchStyle({ outlier_method: e.target.value as any })}
                style={{ width: "100%" }}
              >
                <option value="std">Mean ± k·Std (σ)</option>
                <option value="mad">Median ± k·MAD (robust)</option>
              </select>
            </label>
          </fieldset>
        </div>
      </div>

      {/* Centre: preview */}
      <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12, overflow: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center" }}>
          <b>Preview</b>
          <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
            Preview DPI
            <input
              type="range"
              min={80}
              max={300}
              value={previewDpi}
              onChange={(e) => setPreviewDpi(parseInt(e.target.value, 10))}
            />
            <span style={{ width: 36, textAlign: "right" }}>{previewDpi}</span>
          </label>
        </div>

        {loading && <div style={{ opacity: 0.7, marginTop: 8 }}>Rendering…</div>}
        {error && <pre style={{ whiteSpace: "pre-wrap", color: "crimson" }}>{error}</pre>}

        {issues.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <b>Issues</b>
            <ul>
              {issues.map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ul>
          </div>
        )}

        <div style={{ marginTop: 10 }}>
          {imgSrc ? (
            <img src={imgSrc} style={{ maxWidth: "100%", height: "auto" }} />
          ) : (
            <div>No image yet.</div>
          )}
        </div>
      </div>

      {/* Right: Series and view */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12, minHeight: 0 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
          <h3 style={{ marginTop: 0 }}>Series and view</h3>

          <label>
            Series
            <input
              type="number"
              min={1}
              max={10}
              value={spec.series_count}
              onChange={(e) =>
                setSpec((s) => ({
                  ...s,
                  series_count: clamp(parseInt(e.target.value || "1", 10), 1, 10),
                }))
              }
              style={{ width: "100%" }}
            />
          </label>

          {is3d && (
            <>
              <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
                <label>
                  Elevation{" "}
                  <input
                    type="number"
                    value={viewElev}
                    onChange={(e) => setViewElev(parseFloat(e.target.value || "0"))}
                    style={{ width: "100%" }}
                  />
                </label>
                <label>
                  Azimuth{" "}
                  <input
                    type="number"
                    value={viewAzim}
                    onChange={(e) => setViewAzim(parseFloat(e.target.value || "0"))}
                    style={{ width: "100%" }}
                  />
                </label>
                <label>
                  Roll{" "}
                  <input
                    type="number"
                    value={viewRoll}
                    onChange={(e) => setViewRoll(parseFloat(e.target.value || "0"))}
                    style={{ width: "100%" }}
                  />
                </label>
              </div>

              <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
                <button onClick={() => setPreset("xy")}>XY</button>
                <button onClick={() => setPreset("xz")}>XZ</button>
                <button onClick={() => setPreset("yz")}>YZ</button>
                <button onClick={() => setPreset("reset")}>Reset</button>
              </div>
            </>
          )}
        </div>

        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 10,
            padding: 12,
            overflow: "auto",
            flex: 1,
            minHeight: 0,
          }}
        >
          {spec.series.map((s, i) => (
            <fieldset
              key={i}
              style={{
                border: "1px solid #e5e5e5",
                borderRadius: 10,
                padding: 10,
                marginBottom: 12,
              }}
            >
              <legend style={{ padding: "0 6px" }}>{`Series ${i + 1}`}</legend>

              <label>
                Label
                <input value={s.label} onChange={(e) => patchSeries(i, { label: e.target.value })} style={{ width: "100%" }} />
              </label>

              {/* NEW: input mode toggle */}
              <label style={{ marginTop: 8 }}>
                Input mode
                <select
                  value={s.input_mode}
                  onChange={(e) => patchSeries(i, { input_mode: e.target.value as any })}
                  style={{ width: "100%" }}
                >
                  <option value="inline">Inline (vectors)</option>
                  <option value="table">Table (CSV/TSV)</option>
                </select>
              </label>

              {/* Inline mode */}
              {s.input_mode === "inline" && (
                <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                  {showX && (
                    <label>
                      x
                      <textarea
                        value={s.inline.x_text}
                        onChange={(e) => patchSeriesInline(i, { x_text: e.target.value })}
                        rows={3}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}

                  <label>
                    {valuesOnly ? "values" : "y"}
                    <textarea
                      value={s.inline.y_text}
                      onChange={(e) => patchSeriesInline(i, { y_text: e.target.value })}
                      rows={3}
                      style={{ width: "100%" }}
                    />
                  </label>

                  {is3d && (
                    <label>
                      z
                      <textarea
                        value={s.inline.z_text}
                        onChange={(e) => patchSeriesInline(i, { z_text: e.target.value })}
                        rows={3}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}
                </div>
              )}

              {/* Table mode */}
              {s.input_mode === "table" && (
                <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                  <label>
                    Table data (CSV or TSV, header row recommended)
                    <textarea
                      value={s.inline.table_text}
                      onChange={(e) => patchSeriesInline(i, { table_text: e.target.value })}
                      rows={8}
                      style={{ width: "100%", fontFamily: "monospace" }}
                      placeholder={"x,y\n1,2\n2,4\n3,6\n\nor TSV:\nx\ty\n1\t2\n2\t4"}
                    />
                  </label>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {showX && (
                      <label>
                        x column
                        <input
                          value={s.inline.x_col}
                          onChange={(e) => patchSeriesInline(i, { x_col: e.target.value })}
                          style={{ width: "100%" }}
                        />
                      </label>
                    )}

                    <label>
                      {valuesOnly ? "values column" : "y column"}
                      <input
                        value={s.inline.y_col}
                        onChange={(e) => patchSeriesInline(i, { y_col: e.target.value })}
                        style={{ width: "100%" }}
                      />
                    </label>

                    {is3d && (
                      <label>
                        z column
                        <input
                          value={s.inline.z_col}
                          onChange={(e) => patchSeriesInline(i, { z_col: e.target.value })}
                          style={{ width: "100%" }}
                        />
                      </label>
                    )}
                  </div>
                </div>
              )}

              {/* NEW: std / errorbar toggles (metadata-driven) */}
              {plotTypeMeta?.supports_x_std && (
                <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
                  <input
                    type="checkbox"
                    checked={s.use_x_std}
                    onChange={(e) => patchSeries(i, { use_x_std: e.target.checked })}
                  />
                  Use x std
                </label>
              )}

              {plotTypeMeta?.supports_y_std && (
                <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                  <input
                    type="checkbox"
                    checked={s.use_y_std}
                    onChange={(e) => patchSeries(i, { use_y_std: e.target.checked })}
                  />
                  Use y std
                </label>
              )}

              {/* std fields depend on input mode */}
              {(s.use_x_std || s.use_y_std) && s.input_mode === "inline" && (
                <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                  {s.use_x_std && (
                    <label>
                      x std (inline)
                      <textarea
                        value={s.inline.x_std_text}
                        onChange={(e) => patchSeriesInline(i, { x_std_text: e.target.value })}
                        rows={2}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}
                  {s.use_y_std && (
                    <label>
                      y std (inline)
                      <textarea
                        value={s.inline.y_std_text}
                        onChange={(e) => patchSeriesInline(i, { y_std_text: e.target.value })}
                        rows={2}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}
                </div>
              )}

              {(s.use_x_std || s.use_y_std) && s.input_mode === "table" && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 8 }}>
                  {s.use_x_std && (
                    <label>
                      x std column
                      <input
                        value={s.inline.x_std_col}
                        onChange={(e) => patchSeriesInline(i, { x_std_col: e.target.value })}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}
                  {s.use_y_std && (
                    <label>
                      y std column
                      <input
                        value={s.inline.y_std_col}
                        onChange={(e) => patchSeriesInline(i, { y_std_col: e.target.value })}
                        style={{ width: "100%" }}
                      />
                    </label>
                  )}
                </div>
              )}

              {/* NEW: grouping (table mode only, meta-driven) */}
              {plotTypeMeta?.supports_grouping && s.input_mode === "table" && (
                <>
                  <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
                    <input
                      type="checkbox"
                      checked={s.split_by_group}
                      onChange={(e) => patchSeries(i, { split_by_group: e.target.checked })}
                    />
                    Split by group
                  </label>

                  {s.split_by_group && (
                    <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                      <label>
                        Group column
                        <input
                          value={s.inline.group_col}
                          onChange={(e) => patchSeriesInline(i, { group_col: e.target.value })}
                          style={{ width: "100%" }}
                        />
                      </label>

                      <label>
                        Group order (comma-separated, optional)
                        <input
                          value={s.inline.group_order_text}
                          onChange={(e) => patchSeriesInline(i, { group_order_text: e.target.value })}
                          style={{ width: "100%" }}
                        />
                      </label>
                    </div>
                  )}
                </>
              )}

              <div style={{ marginTop: 10 }}>
                <label>
                  Colour
                  <input
                    value={s.style.color}
                    onChange={(e) => patchSeriesStyle(i, { color: e.target.value })}
                    placeholder="tab:blue or #1f77b4"
                    style={{ width: "100%" }}
                  />
                </label>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
                {plotTypeMeta?.supports_markers !== false && (
                  <label>
                    Marker
                    <select value={s.style.marker} onChange={(e) => patchSeriesStyle(i, { marker: e.target.value })} style={{ width: "100%" }}>
                      {MARKERS.map(([label, code]) => (
                        <option key={code} value={code}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                )}

                {plotTypeMeta?.supports_marker_size !== false && (
                  <label>
                    Marker size
                    <input
                      type="number"
                      step="0.5"
                      min={1}
                      max={40}
                      value={s.style.marker_size}
                      onChange={(e) => patchSeriesStyle(i, { marker_size: parseFloat(e.target.value || "6") })}
                      style={{ width: "100%" }}
                    />
                  </label>
                )}

                {plotTypeMeta?.supports_lines !== false && (
                  <>
                    <label>
                      Line width
                      <input
                        type="number"
                        step="0.1"
                        min={0.1}
                        max={12}
                        value={s.style.line_width}
                        onChange={(e) => patchSeriesStyle(i, { line_width: parseFloat(e.target.value || "1.6") })}
                        style={{ width: "100%" }}
                      />
                    </label>

                    <label>
                      Line style
                      <select value={s.style.line_style} onChange={(e) => patchSeriesStyle(i, { line_style: e.target.value })} style={{ width: "100%" }}>
                        {LINESTYLES.map(([label, code]) => (
                          <option key={code} value={code}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
              </div>

              <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
                <input
                  type="checkbox"
                  checked={s.style.highlight_outliers}
                  onChange={(e) => patchSeriesStyle(i, { highlight_outliers: e.target.checked })}
                />
                Highlight outliers (±kσ)
              </label>
            </fieldset>
          ))}
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => exportFigure("png")}>Export figure</button>
          <button onClick={exportCode}>Export code</button>
          <button onClick={() => exportFigure("pdf")}>PDF</button>
          <button onClick={() => exportFigure("svg")}>SVG</button>
        </div>
      </div>
    </div>
  );
}
