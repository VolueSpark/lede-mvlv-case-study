from plotly.subplots import make_subplots
from flask import Flask, render_template
import plotly.graph_objects as go
import polars as pl
import os, re

PATH = os.path.dirname(__file__)
DATA = os.path.join(PATH, '../lfa/data')
HOST = "0.0.0.0"
PORT = 7070

app = Flask(__name__, template_folder='template')

@app.route('/')
def default():

    neighborhood_list = os.listdir(DATA)
    df = pl.DataFrame(neighborhood_list, schema={'neighborhood_id':pl.Utf8})
    return render_template("table_neighborhood.html",
                           data=df.to_dicts())

@app.route('/neighborhood/<neighborhood_id>')
def neighborhood(neighborhood_id):

    neighborhood_path = os.path.join(DATA, neighborhood_id)
    df = pl.read_parquet(neighborhood_path)
    meter_list = [meter_id for meter_id in df['meter_id'].unique().to_list() if bool(re.match(r'^\d{18}$', meter_id))]
    return render_template("table_meter.html", data=meter_list, neighborhood_id=neighborhood_id)

@app.route('/meter/<neighborhood_id>/<meter_id>')
def meter(neighborhood_id, meter_id):
    neighborhood_path = os.path.join(DATA, neighborhood_id)
    df = pl.read_parquet(neighborhood_path)
         # .filter(pl.col('meter_id') == meter_id))

    df_meter = df.filter(pl.col('meter_id')==meter_id)

    df_sum = df.sort(by='datetime', descending=False).group_by_dynamic('datetime', every='1h').agg(
        pl.col('p_mw').sum().alias('sum_p_kw')*1000,
        pl.col('q_mvar').sum().alias('sum_q_mvar')*1000
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'Active power for meter: {meter_id} and aggregated sum', f'Reactive power for meter: {meter_id} and aggregated sum'],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )

    # active power
    fig.add_trace(go.Scatter(
        x=df_meter['datetime'],
        y=df_meter['p_mw']*1000,
        name='net kWh/h',
        line=dict(color="#89CFF0")),
        row=1,
        col=1,
        secondary_y=False
    )

    fig.add_trace(go.Scatter(
        x=df_sum['datetime'],
        y=df_sum['sum_p_kw'],
        name='net.sum kWh/h',
        line=dict(color="#0000FF")),
        row=1,
        col=1,
        secondary_y=True
    )

    fig.update_yaxes(
        title_text="net kWh/h",
        row=1,
        col=1,
        secondary_y=False,
        title_font=dict(color='#89CFF0')
    )
    fig.update_yaxes(
        title_text="net.sum kWh/h",
        row=1,
        col=1,
        secondary_y=True,
        title_font=dict(color='#0000FF')
    )
    fig.update_xaxes(title_text='time', row=1, col=1)

    # reactive power
    fig.add_trace(go.Scatter(
        x=df_meter['datetime'],
        y=df_meter['q_mvar']*1000,
        name='net kVAr/h',
        line=dict(color="#7393B3")),
        row=2,
        col=1,
        secondary_y=False
    )

    fig.add_trace(go.Scatter(
        x=df_sum['datetime'],
        y=df_sum['sum_q_mvar'],
        name='net.sum kVAr/h',
        line=dict(color="#088F8F")),
        row=2,
        col=1,
        secondary_y=True
    )

    fig.update_yaxes(
        title_text="net kVAr/h",
        row=2,
        col=1,
        secondary_y=False,
        title_font=dict(color='#7393B3')
    )
    fig.update_yaxes(
        title_text="net.sum kVAr/h",
        row=2,
        col=1,
        secondary_y=True,
        title_font=dict(color='#088F8F')
    )
    fig.update_xaxes(title_text='time', row=2, col=1)

    # font details
    fig.update_layout(
        title=dict(text=f"Prosumption profiles for neighourhood: {neighborhood_id}", xanchor='left'),
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