from plotly.subplots import make_subplots
from flask import Flask, render_template, request
import plotly.graph_objects as go
import polars as pl
import os, re

PATH = os.path.dirname(__file__)
METADATA = os.path.join(PATH, '../../data/ami/gold/metadata.parquet')  # meta statistics based on silver data
SILVER_DATA = os.path.join(PATH, '../../data/ami/silver/meas')
TOPOLOGY_DATA = os.path.join(PATH, '../lfa/data') # process data aggregated for each neighborhood for faster processing
HOST = "0.0.0.0"
PORT = 9070

app = Flask(__name__, template_folder='template')

@app.route('/')
def default():

    sort_by = request.args.get('sort_by', default='max_net_p_kwh').lower()

    df = pl.read_parquet(METADATA).sort(by=sort_by, descending=True)
    return render_template("table_metadata.html",
                           data=df.to_dicts())

@app.route('/uuid')
def uuid():

    topology_id = request.args.get('uuid')

    topology_path = os.path.join(TOPOLOGY_DATA, f'{topology_id}.parquet')

    df = pl.read_parquet(topology_path)

    df_sum = df.sort(by='datetime', descending=False).group_by_dynamic('datetime', every='1h').agg(
        pl.col('p_mw').sum().alias('sum_p_kw')*1000,
        pl.col('q_mvar').sum().alias('sum_q_kvar')*1000
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"Aggregated active power for {df.n_unique('meter_id')} meters",
                        f"Aggregated reactive power for {df.n_unique('meter_id')} meters"]
    )

    # active power
    fig.add_trace(go.Scatter(
        x=df_sum['datetime'],
        y=df_sum['sum_p_kw'],
        name='net.sum kWh/h',
        line=dict(color="#89CFF0")),
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="net.sum kWh/h",
        row=1,
        col=1,
        title_font=dict(color='#89CFF0'),
        tickfont=dict(color='#89CFF0')
    )

    fig.update_xaxes(title_text='time', row=1, col=1)

    # reactive power
    fig.add_trace(go.Scatter(
        x=df_sum['datetime'],
        y=df_sum['sum_q_kvar'],
        name='net.sum kVAr/h',
        line=dict(color="#7393B3")),
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="net.sum kVAr/h",
        row=2,
        col=1,
        title_font=dict(color='#7393B3'),
        tickfont=dict(color='#7393B3')
    )

    fig.update_xaxes(title_text='time', row=2, col=1)

    # font details
    fig.update_layout(
        title=dict(text=f"Prosumption profiles for lv topology: {topology_id}", xanchor='left'),
        width=1920*0.9,
        height=1080*0.9,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        )
    )
    return render_template('index.html', plot_div=fig.to_html(full_html=False))

@app.route('/meter')
def meter():

    meter_id = request.args.get('meter_id')
    meter_path = os.path.join(SILVER_DATA, f'{meter_id}')

    df = pl.read_parquet(meter_path)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'Active power [out->grid to consumer; in->consumer to grid]', f'Reactive power [out->grid to consumer; in->consumer to grid]'],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )

    # active power
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['p_kwh_out'],
        name='kWh/h out',
        line=dict(color="#89CFF0")),
        row=1,
        col=1,
        secondary_y=False
    )

    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['p_kwh_in'],
        name='kWh/h in',
        line=dict(color="#0000FF")),
        row=1,
        col=1,
        secondary_y=True
    )

    fig.update_yaxes(
        title_text="kWh/h out",
        row=1,
        col=1,
        secondary_y=False,
        title_font=dict(color='#89CFF0'),
        tickfont=dict(color='#89CFF0')
    )
    fig.update_yaxes(
        title_text="kWh/h in",
        row=1,
        col=1,
        secondary_y=True,
        title_font=dict(color='#0000FF'),
        tickfont=dict(color='#0000FF')
    )
    fig.update_xaxes(title_text='time', row=1, col=1)

    # reactive power
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['q_kvarh_out'],
        name='kVArh/h out',
        line=dict(color="#7393B3")),
        row=2,
        col=1,
        secondary_y=False
    )

    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['q_kvarh_in'],
        name='kVArh/h in',
        line=dict(color="#088F8F")),
        row=2,
        col=1,
        secondary_y=True
    )

    fig.update_yaxes(
        title_text="kVArh/h out",
        row=2,
        col=1,
        secondary_y=False,
        title_font=dict(color='#7393B3'),
        tickfont=dict(color='#7393B3')
    )
    fig.update_yaxes(
        title_text="kVArh/h in",
        row=2,
        col=1,
        secondary_y=True,
        title_font=dict(color='#088F8F'),
        tickfont=dict(color='#088F8F')
    )
    fig.update_xaxes(title_text='time', row=2, col=1)

    # font details
    fig.update_layout(
        title=dict(text=f"Prosumption profiles meter: {meter_id}", xanchor='left'),
        width=1920*0.9,
        height=1080*0.9,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        )
    )
    return render_template('index.html', plot_div=fig.to_html(full_html=False))


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)