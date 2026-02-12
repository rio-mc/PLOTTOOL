export type FamilyItem = { value: string; label: string };

export type ControlKind = "bool" | "int" | "float" | "choice" | "text";

export type ControlSpec = {
  key: string;
  label?: string;
  kind: ControlKind;

  default?: any;
  min?: number;
  max?: number;
  step?: number;

  choices?: Array<{ label: string; value: any }> | any[];

  help?: string;
};

export type PlotTypeMeta = {
  family: string;
  label: string;

  requires_x: boolean;
  requires_y: boolean;
  y_is_values_only: boolean;
  requires_z: boolean;

  supports_markers: boolean;
  supports_marker_size: boolean;
  supports_lines: boolean;
  supports_grouping: boolean;
  supports_errorbars: boolean;

  supports_log_x: boolean;
  supports_log_y: boolean;

  requires_xy: boolean;
  supports_x_std: boolean;
  supports_y_std: boolean;

  x_is_datetime: boolean;

  controls: ControlSpec[];
};

export type PlotTypeItem = { value: string; label: string; meta: PlotTypeMeta };

export type RenderJsonResponse = {
  mime: string;
  format: string;
  payload_base64: string;
  issues?: string[];
};
