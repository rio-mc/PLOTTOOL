export type SeriesInputMode = "inline" | "table";

export type SeriesInlineData = {
  // inline vectors
  x_text: string;
  y_text: string;
  z_text: string;

  // optional std vectors for inline mode
  x_std_text: string;
  y_std_text: string;

  // table mode
  table_text: string;

  // column names (table mode)
  x_col: string;
  y_col: string;
  z_col: string;

  // std columns (optional)
  x_std_col: string;
  y_std_col: string;

  // grouping (optional)
  group_col: string;
  group_order_text: string;
};

export type SeriesStyleSpec = {
  color: string;
  marker: string;
  marker_size: number;
  line_width: number;
  line_style: string; // solid|dashed|dotted|dashdot
  highlight_outliers: boolean;
};

export type SeriesSpec = {
  label: string;

  input_mode: SeriesInputMode;

  use_x_std: boolean;
  use_y_std: boolean;

  split_by_group: boolean;
  group_label_prefix: string;
  group_color_by_cycle: boolean;

  inline: SeriesInlineData;
  style: SeriesStyleSpec;
};

export type StyleSpec = {
  font_family: string;

  title: string;
  title_bold: boolean;
  title_italic: boolean;
  title_underline: boolean;
  title_offset: number | null;

  x_label: string;
  y_label: string;
  z_label: string | null;

  x_tick_label_angle: number;
  y_tick_label_angle: number;

  show_grid: boolean;
  show_minor_ticks: boolean;
  show_minor_grid: boolean;
  show_legend: boolean;

  base_font_size: number;
  title_font_size: number;

  outlier_sigma: number;
  outlier_method: "std" | "mad";
};

export type PlotSettingsSpec = {
  options: Record<string, any>;
};

export type PlotSpec = {
  plot_family: string;
  plot_type: string;
  settings: PlotSettingsSpec;
  you_got_data: boolean;
  series_count: number;
  series: SeriesSpec[];
  style: StyleSpec;
};
