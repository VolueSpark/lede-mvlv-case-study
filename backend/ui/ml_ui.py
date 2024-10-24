from plotly.subplots import make_subplots
from flask import Flask, render_template, request
import plotly.graph_objects as go
import polars as pl
import os, re

PATH = os.path.dirname(__file__)


UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
#UUID ='d1e83c9d-caa9-5bf0-89df-4238df995d06'

DATA = os.path.join(PATH, f'../ml/{UUID}/data/gold/data.parquet')
METADATA = os.path.join(PATH, f'../ml/{UUID}/data/gold/metadata.parquet')

HOST = "0.0.0.0"
PORT = 9080

app = Flask(__name__, template_folder='template')

@app.route('/')
def default():

    df = pl.read_parquet(METADATA)
    return render_template("table_prediction.html",
                           data=df.to_dicts(), uuid=UUID)

@app.route('/meter')
def meter():

    data = pl.read_parquet(DATA)

    aed_id = request.args.get('id')
    k = int(request.args.get('k', default=1))-1

    offset = k+data.filter(type='target')['offset'].min()
    columns = ['base']+data.select(pl.col(f'^X_.*{aed_id}.*$')).columns
    target =data.filter((pl.col('type')=='target') & (pl.col('offset')==offset)).select(columns)
    forecast = data.filter((pl.col('type')=='forecast') & (pl.col('offset')==offset)).select(columns)

    #data = data.select(['base', 'offset', 'type']+data.select(pl.col(f'^X_.*{meter_id}.*$')).columns).filter((pl.col('type')=='forecast') | (pl.col('type')=='target'))

    n = len(target.select(r'^X_.*$').columns)
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles= target.select(r'^X_.*$').columns,
    )

    for i, meter_id in enumerate(target.select(r'^X_.*$').columns):

        fig.add_trace(go.Scatter(
            x=target['base'],
            y=target[meter_id],
            name='target',
            line=dict(color="#89CFF0")),
            row=i+1,
            col=1
        )

        fig.add_trace(go.Scatter(
            x=forecast['base'],
            y=forecast[meter_id],
            name='prediction',
            line=dict(color="#7393B3",dash="dash")),
            row=i+1,
            col=1
        )

        fig.update_xaxes(title_text=f'k={k}', row=i+1, col=1)

        # font details
        fig.update_layout(
            width=1920*0.9,
            height=1080*0.9,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="#000000"
            )
        )
        fig.update_annotations(font_size=28)


    return render_template('index.html', plot_div=fig.to_html(full_html=False))

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)