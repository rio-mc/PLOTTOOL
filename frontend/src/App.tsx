import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "http://127.0.0.1:8000";

type FamilyItem = { value: string; label: string };
type PlotTypeItem = { value: string; label: string; meta?: any };
type RenderJsonResponse = { mime: string; format: string; payload_base64: string; issues?: string[] };

type SeriesState = {
  label: string;
  x_text: string;
  y_text: string;
  z_text: string;
  color: string;
  marker: string;
  marker_size: number;
  line_width: number;
  line_style: string;
  highlight_outliers: boolean;
};

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

function defaultSeries(i: number): SeriesState {
  return {
    label: `Series ${i + 1}`,
    x_text: "1, 2, 3",
    y_text: "2, 4, 6",
    z_text: "",
    color: "",
    marker: "o",
    marker_size: 6.0,
    line_width: 1.6,
    line_style: "solid",
    highlight_outliers: false,
  };
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

  // ---- spec-like state (matches desktop fields) ----
  const [plotFamily, setPlotFamily] = useState<string>("basic");
  const [plotType, setPlotType] = useState<string>("line");
  const [plotTypeMeta, setPlotTypeMeta] = useState<any>(null);

  const [seriesCount, setSeriesCount] = useState<number>(1);
  const [series, setSeries] = useState<SeriesState[]>([defaultSeries(0)]);

  // style panel
  const [fontFamily, setFontFamily] = useState("Times New Roman");
  const [title, setTitle] = useState("My Title");
  const [titleBold, setTitleBold] = useState(true);
  const [titleItalic, setTitleItalic] = useState(false);
  const [titleUnderline, setTitleUnderline] = useState(false);
  const [xLabel, setXLabel] = useState("X axis");
  const [yLabel, setYLabel] = useState("Y axis");
  const [gridMajor, setGridMajor] = useState(true);
  const [minorTicks, setMinorTicks] = useState(false);
  const [minorGrid, setMinorGrid] = useState(false);
  const [legend, setLegend] = useState(true);
  const [baseFontSize, setBaseFontSize] = useState(11);
  const [titleFontSize, setTitleFontSize] = useState(13);
  const [xTickAngle, setXTickAngle] = useState(0);
  const [yTickAngle, setYTickAngle] = useState(0);
  const [outlierK, setOutlierK] = useState(3.0);
  const [outlierMethod, setOutlierMethod] = useState<"std" | "mad">("std");

  // 3D view controls
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
        // keep default if present, else first
        if (fam.length && !fam.find((f) => f.value === plotFamily)) setPlotFamily(fam[0].value);

        const fnts = await getJson<string[]>("/meta/fonts");
        setFonts(fnts);
        if (fnts.includes("Times New Roman")) setFontFamily("Times New Roman");
        else if (fnts.length) setFontFamily(fnts[0]);
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
        const t = await getJson<PlotTypeItem[]>(`/meta/types?family=${encodeURIComponent(plotFamily)}`);
        setTypes(t);
        // snap to first if current not present
        if (t.length && !t.find((x) => x.value === plotType)) setPlotType(t[0].value);
      } catch (e: any) {
        setError(e?.message ?? String(e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotFamily]);

  // ---- Update plotTypeMeta whenever plotType changes ----
  useEffect(() => {
    const item = types.find((t) => t.value === plotType);
    setPlotTypeMeta(item?.meta ?? null);
  }, [plotType, types]);

  const is3d = !!plotTypeMeta?.requires_z;
  const showX = plotTypeMeta?.requires_x !== false;
  const valuesOnly = !!plotTypeMeta?.y_is_values_only || !showX;

  // ---- Keep series array sized to seriesCount (like your Qt logic) ----
  useEffect(() => {
    setSeries((prev) => {
      const next = [...prev];
      while (next.length < seriesCount) next.push(defaultSeries(next.length));
      while (next.length > seriesCount) next.pop();
      return next;
    });
  }, [seriesCount]);

  // ---- Spec JSON that backend expects (matches PlotSpec/SeriesSpec/StyleSpec) ----
  const spec = useMemo(() => {
    return {
      plot_family: plotFamily,
      plot_type: plotType,
      you_got_data: true,
      series_count: seriesCount,
      series: series.map((s) => ({
        label: s.label,
        inline: {
          x_text: s.x_text,
          y_text: s.y_text,
          z_text: s.z_text,
        },
        style: {
          color: s.color,
          marker: s.marker,
          marker_size: s.marker_size,
          line_width: s.line_width,
          line_style: s.line_style,
          highlight_outliers: s.highlight_outliers,
        },
      })),
      style: {
        font_family: fontFamily,
        title,
        title_bold: titleBold,
        title_italic: titleItalic,
        title_underline: titleUnderline,
        x_label: xLabel,
        y_label: yLabel,
        x_tick_label_angle: xTickAngle,
        y_tick_label_angle: yTickAngle,
        show_grid: gridMajor,
        show_minor_ticks: minorTicks,
        show_minor_grid: minorGrid,
        show_legend: legend,
        base_font_size: baseFontSize,
        title_font_size: titleFontSize,
        outlier_sigma: outlierK,
        outlier_method: outlierMethod,
      },
    };
  }, [
    plotFamily,
    plotType,
    seriesCount,
    series,
    fontFamily,
    title,
    titleBold,
    titleItalic,
    titleUnderline,
    xLabel,
    yLabel,
    xTickAngle,
    yTickAngle,
    gridMajor,
    minorTicks,
    minorGrid,
    legend,
    baseFontSize,
    titleFontSize,
    outlierK,
    outlierMethod,
  ]);

  async function renderPreview() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/render_json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
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

  // Debounced like Qt (120ms) — keep it snappy
  useEffect(() => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => {
      renderPreview();
    }, 140);
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, previewDpi, viewElev, viewAzim, viewRoll]);

  async function exportFigure(format: "png" | "pdf" | "svg") {
    const dpi = format === "png" ? 300 : undefined;
    const res = await fetch(`${API_BASE}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spec,
        format,
        dpi: dpi ?? 300,
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

  function updateSeries(i: number, patch: Partial<SeriesState>) {
    setSeries((prev) => prev.map((s, idx) => (idx === i ? { ...s, ...patch } : s)));
  }

  function setPreset(p: "reset" | "xy" | "xz" | "yz") {
    if (p === "reset") {
      setViewElev(30); setViewAzim(-60); setViewRoll(0);
    } else if (p === "xy") {
      setViewElev(90); setViewAzim(-90); setViewRoll(0);
    } else if (p === "xz") {
      setViewElev(0); setViewAzim(-90); setViewRoll(0);
    } else if (p === "yz") {
      setViewElev(0); setViewAzim(0); setViewRoll(0);
    }
  }

  // ---- Layout: mimic QSplitter (left controls / centre preview / right panel) ----
  return (
    <div style={{ height: "100vh", display: "grid", gridTemplateColumns: "380px 1fr 420px", gap: 12, padding: 12 }}>
      {/* Left: controls */}
      <div style={{ overflow: "auto", border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
        <h3 style={{ marginTop: 0 }}>Controls</h3>

        <div style={{ display: "grid", gap: 10 }}>
          <label>
            Plot family
            <select value={plotFamily} onChange={(e) => setPlotFamily(e.target.value)} style={{ width: "100%" }}>
              {families.map((f) => (
                <option key={f.value} value={f.value}>{f.label}</option>
              ))}
            </select>
          </label>

          <label>
            Plot type
            <select value={plotType} onChange={(e) => setPlotType(e.target.value)} style={{ width: "100%" }}>
              {types.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </label>

          <fieldset style={{ border: "1px solid #e5e5e5", borderRadius: 10, padding: 10 }}>
            <legend style={{ padding: "0 6px" }}>Style</legend>

            <label>
              Font
              <input
                list="fonts"
                value={fontFamily}
                onChange={(e) => setFontFamily(e.target.value)}
                style={{ width: "100%" }}
              />
              <datalist id="fonts">
                {fonts.map((f) => <option key={f} value={f} />)}
              </datalist>
            </label>

            <div style={{ marginTop: 10 }}>
              <div>Title</div>
              <div style={{ display: "flex", gap: 6 }}>
                <input value={title} onChange={(e) => setTitle(e.target.value)} style={{ flex: 1 }} />
                <button onClick={() => setTitleBold((v) => !v)} style={{ fontWeight: "bold", opacity: titleBold ? 1 : 0.5 }}>B</button>
                <button onClick={() => setTitleItalic((v) => !v)} style={{ fontStyle: "italic", opacity: titleItalic ? 1 : 0.5 }}>I</button>
                <button onClick={() => setTitleUnderline((v) => !v)} style={{ textDecoration: "underline", opacity: titleUnderline ? 1 : 0.5 }}>U</button>
              </div>
            </div>

            <label style={{ marginTop: 10 }}>
              x label
              <input value={xLabel} onChange={(e) => setXLabel(e.target.value)} style={{ width: "100%" }} />
            </label>

            <label>
              y label
              <input value={yLabel} onChange={(e) => setYLabel(e.target.value)} style={{ width: "100%" }} />
            </label>

            <label>
              x tick label angle
              <input type="number" value={xTickAngle} onChange={(e) => setXTickAngle(clamp(parseInt(e.target.value || "0", 10), -180, 180))} style={{ width: "100%" }} />
            </label>

            <label>
              y tick label angle
              <input type="number" value={yTickAngle} onChange={(e) => setYTickAngle(clamp(parseInt(e.target.value || "0", 10), -180, 180))} style={{ width: "100%" }} />
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
              <input type="checkbox" checked={gridMajor} onChange={(e) => setGridMajor(e.target.checked)} />
              Grid (major)
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={minorTicks}
                onChange={(e) => {
                  setMinorTicks(e.target.checked);
                  if (!e.target.checked) setMinorGrid(false);
                }}
              />
              Minor ticks
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center", opacity: minorTicks ? 1 : 0.5 }}>
              <input
                type="checkbox"
                checked={minorGrid}
                disabled={!minorTicks}
                onChange={(e) => setMinorGrid(e.target.checked)}
              />
              Grid (minor)
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input type="checkbox" checked={legend} onChange={(e) => setLegend(e.target.checked)} />
              Legend
            </label>

            <label>
              Text font size
              <input type="number" value={baseFontSize} onChange={(e) => setBaseFontSize(clamp(parseInt(e.target.value || "11", 10), 7, 36))} style={{ width: "100%" }} />
            </label>

            <label>
              Title font size
              <input type="number" value={titleFontSize} onChange={(e) => setTitleFontSize(clamp(parseInt(e.target.value || "13", 10), 7, 48))} style={{ width: "100%" }} />
            </label>

            <label>
              Outlier k
              <input type="number" step="0.25" value={outlierK} onChange={(e) => setOutlierK(parseFloat(e.target.value || "3"))} style={{ width: "100%" }} />
            </label>

            <label>
              Outlier method
              <select value={outlierMethod} onChange={(e) => setOutlierMethod(e.target.value as any)} style={{ width: "100%" }}>
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
            <input type="range" min={80} max={300} value={previewDpi} onChange={(e) => setPreviewDpi(parseInt(e.target.value, 10))} />
            <span style={{ width: 36, textAlign: "right" }}>{previewDpi}</span>
          </label>
        </div>

        {loading && <div style={{ opacity: 0.7, marginTop: 8 }}>Rendering…</div>}
        {error && <pre style={{ whiteSpace: "pre-wrap", color: "crimson" }}>{error}</pre>}

        {issues.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <b>Issues</b>
            <ul>
              {issues.map((x, i) => <li key={i}>{x}</li>)}
            </ul>
          </div>
        )}

        <div style={{ marginTop: 10 }}>
          {imgSrc ? <img src={imgSrc} style={{ maxWidth: "100%", height: "auto" }} /> : <div>No image yet.</div>}
        </div>
      </div>

      {/* Right: “Series and view” + series scroll + export buttons */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12, minHeight: 0 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12 }}>
          <h3 style={{ marginTop: 0 }}>Series and view</h3>

          <label>
            Series
            <input
              type="number"
              min={1}
              max={10}
              value={seriesCount}
              onChange={(e) => setSeriesCount(clamp(parseInt(e.target.value || "1", 10), 1, 10))}
              style={{ width: "100%" }}
            />
          </label>

          {is3d && (
            <>
              <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
                <label>Elevation <input type="number" value={viewElev} onChange={(e) => setViewElev(parseFloat(e.target.value || "0"))} style={{ width: "100%" }} /></label>
                <label>Azimuth <input type="number" value={viewAzim} onChange={(e) => setViewAzim(parseFloat(e.target.value || "0"))} style={{ width: "100%" }} /></label>
                <label>Roll <input type="number" value={viewRoll} onChange={(e) => setViewRoll(parseFloat(e.target.value || "0"))} style={{ width: "100%" }} /></label>
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

        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12, overflow: "auto", flex: 1, minHeight: 0 }}>
          {series.map((s, i) => (
            <fieldset key={i} style={{ border: "1px solid #e5e5e5", borderRadius: 10, padding: 10, marginBottom: 12 }}>
              <legend style={{ padding: "0 6px" }}>{`Series ${i + 1}`}</legend>

              <label>
                Label
                <input value={s.label} onChange={(e) => updateSeries(i, { label: e.target.value })} style={{ width: "100%" }} />
              </label>

              <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                {showX && (
                  <label>
                    x
                    <textarea value={s.x_text} onChange={(e) => updateSeries(i, { x_text: e.target.value })} rows={3} style={{ width: "100%" }} />
                  </label>
                )}

                <label>
                  {valuesOnly ? "values" : "y"}
                  <textarea value={s.y_text} onChange={(e) => updateSeries(i, { y_text: e.target.value })} rows={3} style={{ width: "100%" }} />
                </label>

                {is3d && (
                  <label>
                    z
                    <textarea value={s.z_text} onChange={(e) => updateSeries(i, { z_text: e.target.value })} rows={3} style={{ width: "100%" }} />
                  </label>
                )}
              </div>

              <div style={{ marginTop: 10 }}>
                <label>
                  Colour
                  <input value={s.color} onChange={(e) => updateSeries(i, { color: e.target.value })} placeholder="tab:blue or #1f77b4" style={{ width: "100%" }} />
                </label>
              </div>

              {/* Style controls - mirror apply_plot_type visibility */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
                {plotTypeMeta?.supports_markers !== false && (
                  <label>
                    Marker
                    <select value={s.marker} onChange={(e) => updateSeries(i, { marker: e.target.value })} style={{ width: "100%" }}>
                      {MARKERS.map(([label, code]) => (
                        <option key={code} value={code}>{label}</option>
                      ))}
                    </select>
                  </label>
                )}

                {plotTypeMeta?.supports_marker_size !== false && (
                  <label>
                    Marker size
                    <input type="number" step="0.5" min={1} max={40} value={s.marker_size}
                      onChange={(e) => updateSeries(i, { marker_size: parseFloat(e.target.value || "6") })}
                      style={{ width: "100%" }}
                    />
                  </label>
                )}

                {plotTypeMeta?.supports_lines !== false && (
                  <>
                    <label>
                      Line width
                      <input type="number" step="0.1" min={0.1} max={12} value={s.line_width}
                        onChange={(e) => updateSeries(i, { line_width: parseFloat(e.target.value || "1.6") })}
                        style={{ width: "100%" }}
                      />
                    </label>

                    <label>
                      Line style
                      <select value={s.line_style} onChange={(e) => updateSeries(i, { line_style: e.target.value })} style={{ width: "100%" }}>
                        {LINESTYLES.map(([label, code]) => (
                          <option key={code} value={code}>{label}</option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
              </div>

              <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
                <input
                  type="checkbox"
                  checked={s.highlight_outliers}
                  onChange={(e) => updateSeries(i, { highlight_outliers: e.target.checked })}
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
