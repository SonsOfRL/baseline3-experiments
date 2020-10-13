import os
import pandas as pd
import dash
import argparse
from collections import defaultdict
import warnings
import numpy as np

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_daq as daq
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

from lab_sb3.dashboard.reader import (read_logs,
                                      save_experiment_data,
                                      load_experiment_data)


def column_operation_layouts(column_names):
    return [
        html.Div(
            id="columnOpSelect",
            className="",
            children=[
                html.H6("Column Operation", className="pretty_container_head"),
                *[dcc.RadioItems(
                    id="_".join(
                        ["columnOp_{}".format(column_name)]),
                    options=[
                        {"label": "max", "value": "max"},
                        {"label": "last", "value": "last"},
                        {"label": "mean", "value": "mean"},
                        {"label": "sum", "value": "sum"},
                    ],
                    value="last",
                    labelStyle={"display": "inline-block"})
                  for ix, column_name in enumerate(column_names)]
            ]
        ),
        html.Div(
            id="trialOpSelect",
            className="",
            children=[
                html.H6("Trial Operation", className="pretty_container_head"),
                *[dcc.RadioItems(
                    id="_".join(
                        ["trialOp_{}".format(column_name)]),
                    options=[
                        {"label": "max", "value": "max"},
                        {"label": "mean", "value": "mean"},
                        {"label": "sum", "value": "sum"},
                    ],
                    value="mean",
                    labelStyle={"display": "inline-block"})
                  for ix, column_name in enumerate(column_names)]
            ]
        )
    ]


def column_name_layout(column_names):
    return html.Div(
        id="columnSelectBox",
        className="",
        children=[
            html.H6("Column Name", className="pretty_container_head"),
            dcc.Checklist(
                id="columnNameSelect",
                # className="",
                options=[
                    {"label": name, "value": name}
                    for name in column_names
                ],
            )
        ]
    )


def process_dataframe(values, column_op):
    if column_op == "last":
        return values.iloc[-1].item()
    if column_op == "mean":
        return values.mean()
    if column_op == "sum":
        return values.sum()
    if column_op == "max":
        return values.max()
    else:
        raise RuntimeError("ColumnOp could not found")


def trial_op_fn(trial_op):
    if trial_op == "max":
        return max
    if trial_op == "min":
        return min
    if trial_op == "sum":
        return sum
    if trial_op == "mean":
        return lambda x: sum(x)/len(x)
    else:
        raise RuntimeError("TrialOp could not found")


def prepare_dataframe(trial_dict, column_op_map):
    row_items = []
    for trial_name, trial_dict in trial_dict.items():

        trial_values = defaultdict(list)
        for df in trial_dict["data"]:
            for col_name, col_op_dict in column_op_map.items():
                trial_values[col_name].append(process_dataframe(
                    df[col_name].dropna(), col_op_dict["columnop"]))

        for key, values in trial_values.items():
            trial_values[key] = trial_op_fn(col_op_dict["trialop"])(values)
        trial_values["Trial"] = trial_name

        row_items.append(trial_values)
    return pd.DataFrame.from_records(row_items)


def merge_frames(frames, xlabel):
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, how="outer", sort=True, on=xlabel)
    return merged.fillna(method="pad").fillna(method="bfill")


def quantile_merged_frames(merged_frames, xlabel, q_up, q_down, interpolation="linear"):
    merged_frames = merged_frames.set_index(xlabel)
    upper_frames = merged_frames.quantile(
        q=q_up, axis=1, interpolation=interpolation)
    lower_frames = merged_frames.quantile(
        q=q_down, axis=1, interpolation=interpolation)
    return upper_frames, lower_frames


def fill_between(upper_frames, lower_frames, **kwargs):
    return go.Scatter(
        x=np.concatenate([upper_frames.index, lower_frames.index[::-1]]),
        y=pd.concat([upper_frames, lower_frames[::-1]]),
        fill="toself",
        hoveron="points",
        **kwargs
    )


def smoother(frame, alpha):
    return frame.ewm(alpha=alpha).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dashboard")
    parser.add_argument(
        "--logs-dir", help="Directory of logs", type=str, action="store")

    cl_args = parser.parse_args()

    exp_data = {log_dir: read_logs(log_dir)
                for log_dir in [cl_args.logs_dir]}

    experiment_names = [{"label": key, "value": key}
                        for key in exp_data.keys()]

    app = dash.Dash(__name__)
    app.layout = html.Div(id="mainContainer", children=[

        html.Div(id="header",
                 className="pretty_container row flex-display",
                 style={"margin-bottom": "25px"},
                 children=[
                     html.Div(id="title",
                              className="three column",
                              children=[
                                  html.H2("Pavlov's Watchdog")
                              ]),
                 ]),

        html.Div(id="contentSelect",
                 className="pretty_container twelve columns",
                 children=[
                     dcc.RadioItems(
                         options=[
                             {"label": "Line Plots", "value": "line"},
                             {"label": "Parameter Correlation", "value": "corr"},
                         ],
                         value="line",
                         labelStyle={"display": "inline-block"})
                 ]),

        html.Div(id="plots",
                 className="row flex-display",
                 children=[
                     html.Div(id="axisSelect",
                              className="pretty_container two columns",
                              children=[
                                  html.H6("Axis Select",
                                          className="pretty_container_head"),
                                  html.P("Y axis"),
                                  dcc.RadioItems(
                                      id="yAxisSelectRadial",
                                  ),
                                  html.P("X axis"),
                                  dcc.RadioItems(
                                      id="xAxisSelectRadial",
                                  )
                              ]),
                     html.Div(
                         className="pretty_container nine columns",
                         children=[
                             html.Div(
                                 className="row flex-display",
                                 children=[
                                     html.Div(
                                         className="mini_container flex-three",
                                         children=[
                                             html.H6("Smoothing"),
                                             dcc.Slider(
                                                 id="lineSmoothSlider",
                                                 min=0.01,
                                                 max=1,
                                                 step=0.01,
                                                 value=0.01,
                                             ),
                                         ],
                                     ),
                                     html.Div(
                                         className="mini_container flex-two",
                                         children=[
                                             html.H6("Quantile Method"),
                                             dcc.Dropdown(
                                                 id="lineInterpolation",
                                                 options=[
                                                    {"label": "linear",
                                                        "value": "linear"},
                                                    {"label": "lower",
                                                        "value": "lower"},
                                                    {"label": "higher",
                                                        "value": "higher"},
                                                    {"label": "midpoint",
                                                        "value": "midpoint"},
                                                    {"label": "nearest",
                                                        "value": "nearest"},
                                                 ],
                                                 value="linear",
                                             ),
                                         ],
                                     ),
                                     html.Div(
                                         className="mini_container flex-three",
                                         children=[
                                             html.H6("Quantile Range"),
                                             dcc.RangeSlider(
                                                 id="lineQuantileSlider",
                                                 min=0,
                                                 max=1,
                                                 step=0.05,
                                                 value=[0.25, 0.75]
                                             ),
                                         ],
                                     ),
                                     html.Div(
                                         className="mini_container flex-three",
                                         children=[
                                             html.H6("Quantile Area Plot"),
                                             daq.BooleanSwitch(
                                                 id="QuantilePlotSwitch",
                                                 on=True
                                             ),
                                         ],
                                     ),
                                 ]
                             ),
                             dcc.Graph(id="main_graph")
                         ],
                     ),
                 ]),


        html.Div(id="dataSelect",
                 #  className="pretty_container",
                 children=[
                     html.Div(id="experimentSelect",
                              className="pretty_container two columns",
                              children=[
                                 html.H6("Experiments",
                                         className="pretty_container_head"),
                                 dcc.Checklist(
                                     id="experimentSelectChecklist",
                                     options=experiment_names
                                 ),
                              ]),

                    html.Div(id="trialParams",
                             className="pretty_container eight columns",
                             children=[
                                 html.H6("Select Experiment"),
                                 dcc.Dropdown(
                                     id="trialDataFrameSelect",
                                     options=experiment_names
                                 ),
                                 html.H6("Hyperparameters"),

                                 dash_table.DataTable(
                                     id="trialParamFrame",
                                     page_action="none",
                                     filter_action="native",
                                     sort_action="native",
                                     fixed_rows={"headers": True},
                                     style_cell={
                                         "minWidth": 95, "maxWidth": 95, "width": 95
                                     }
                                 )
                             ]),
                    html.Div(id="columnSelect",
                             className="pretty_container",
                             ),
                     html.Div(id="trialData",
                              className="pretty_container six columns",
                              children=[
                                  html.H6("Trials"),

                                  dash_table.DataTable(
                                      id="trialDataFrame",
                                      page_action="none",
                                      filter_action="native",
                                      sort_action="native",
                                      fixed_rows={"headers": True},
                                      style_cell={
                                         "minWidth": 95, "maxWidth": 95, "width": 95
                                      }
                                  )
                              ]),


                 ]),


    ])

    @app.callback(
        [Output("columnSelect", "children"),
         Output("yAxisSelectRadial", "options"),
         Output("xAxisSelectRadial", "options")],
        Input("trialDataFrameSelect", "value"),
        State("columnSelect", "children")
    )
    def update_column_select(selected_exp, children):

        if selected_exp is None:
            raise PreventUpdate

        column_names = set()
        for key, trial_data in exp_data[selected_exp].items():
            for df in trial_data["data"]:
                column_names = column_names.union(set(df.columns))

        return ([column_name_layout(column_names),
                 *column_operation_layouts(column_names)],
                [{"label": name, "value": name} for name in column_names],
                [{"label": name, "value": name} for name in column_names])

    @app.callback(
        [Output("trialParamFrame", "data"),
         Output("trialParamFrame", "columns")],
        [Input("trialDataFrameSelect", "value")],
    )
    def update_param_list(selected_exp):
        if selected_exp is None:
            raise PreventUpdate

        trial_info = exp_data[selected_exp]
        trial_param_frame = pd.DataFrame.from_records(
            [{"Trial": trial_name, **info["hyperparameters"]}
             for trial_name, info in trial_info.items()]
        )

        if trial_param_frame.size == 0:
            raise PreventUpdate

        return (
            trial_param_frame.to_dict("records"),
            [{"id": key, "name": key} for key in trial_param_frame.columns],
        )

    @app.callback(
        [Output("trialDataFrame", "data"),
         Output("trialDataFrame", "columns")],
        [Input("trialDataFrameSelect", "value"),
         Input("columnNameSelect", "value")],
        [State("columnOpSelect", "children"),
         State("trialOpSelect", "children")]
    )
    def update_experiment_list(selected_exp, selected_columns, column_ops, trial_ops):
        if selected_exp is None:
            raise PreventUpdate

        if selected_columns is None:
            raise PreventUpdate

        column_op_map = {}
        for colop, triop in zip(column_ops[1:], trial_ops[1:]):
            name = colop["props"]["id"].partition("columnOp_")[2]
            if name in selected_columns:
                column_op_map[name] = {"columnop": colop["props"]["value"],
                                       "trialop": triop["props"]["value"]}

        df = prepare_dataframe(exp_data[selected_exp], column_op_map)
        return df.to_dict("records"), [{"id": c, "name": c} for c in reversed(df.columns)]

    @app.callback(
        Output("main_graph", "figure"),
        [Input("trialDataFrame", "derived_virtual_data"),
         Input("xAxisSelectRadial", "value"),
         Input("yAxisSelectRadial", "value"),
         Input("lineInterpolation", "value"),
         Input("lineQuantileSlider", "value"),
         Input("QuantilePlotSwitch", "on"),
         Input("lineSmoothSlider", "value")],
        [State("trialDataFrameSelect", "value")]
    )
    def update_plot(trials, xlabel, ylabel, interpolation, quantiles, is_area_plot, smooth_ratio, exp_name):
        if exp_name is None:
            raise PreventUpdate

        trial_data = exp_data[exp_name]

        if xlabel is None:
            raise PreventUpdate

        if ylabel is None:
            raise PreventUpdate

        if len(trials) == 0:
            raise PreventUpdate

        if exp_name is None:
            raise PreventUpdate

        smooth_ratio = 1 - smooth_ratio

        fig = go.Figure()
        for trial_info in trials:
            trial_name = trial_info["Trial"]
            frames = [df[[xlabel, ylabel]]
                      for df in trial_data[trial_name]["data"]]

            if len(frames) < 2 or is_area_plot is False:
                if len(frames) == 0:
                    continue
                for ix, frame in enumerate(frames):
                    fig.add_trace(
                        go.Scatter(
                            x=frame[xlabel],
                            y=smoother(frame[ylabel], smooth_ratio),
                            name="_".join([trial_name, str(ix)]),
                            line=dict(shape="spline", smoothing=1.3, width=2)
                        )
                    )
            else:
                merged_frames = merge_frames(frames, xlabel)
                upper_frames, lower_frames = [smoother(df, smooth_ratio)
                                              for df in quantile_merged_frames(
                    merged_frames, xlabel, *quantiles, interpolation=interpolation)]
                fig.add_trace(fill_between(
                    upper_frames, lower_frames, name=trial_name, line=dict(shape="spline", smoothing=1.3, width=2)))

        fig.update_layout(
            dict(
                autosize=False,
                height=600,
                plot_bgcolor="white",
            )
        )
        fig.update_xaxes(showgrid=True, gridwidth=1,
                         gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=1,
                         gridcolor="lightgrey")
        return fig

    app.run_server(debug=True)
