import type { PlotSpec, SeriesSpec } from "./types";

export function defaultSeries(i: number): SeriesSpec {
  return {
    label: `Series ${i + 1}`,

    input_mode: "inline",

    use_x_std: false,
    use_y_std: false,

    split_by_group: false,
    group_label_prefix: "",
    group_color_by_cycle: true,

    inline: {
      x_text: "1, 2, 3",
      y_text: "2, 4, 6",
      z_text: "",

      x_std_text: "",
      y_std_text: "",

      table_text: "",

      x_col: "x",
      y_col: "y",
      z_col: "z",

      x_std_col: "x_std",
      y_std_col: "y_std",

      group_col: "group",
      group_order_text: "",
    },

    style: {
      color: "",
      marker: "o",
      marker_size: 6.0,
      line_width: 1.6,
      line_style: "solid",
      highlight_outliers: false,
    },
  };
}

export function defaultSpec(): PlotSpec {
  return {
    plot_family: "basic",
    plot_type: "line",

    settings: { options: {} },

    you_got_data: true,
    series_count: 1,
    series: [defaultSeries(0)],

    style: {
      font_family: "Times New Roman",

      title: "My Title",
      title_bold: true,
      title_italic: false,
      title_underline: false,
      title_offset: null,

      x_label: "X axis",
      y_label: "Y axis",
      z_label: null,

      x_tick_label_angle: 0,
      y_tick_label_angle: 0,

      show_grid: true,
      show_minor_ticks: false,
      show_minor_grid: false,
      show_legend: true,

      base_font_size: 11,
      title_font_size: 13,

      outlier_sigma: 3.0,
      outlier_method: "std",
    },
  };
}
