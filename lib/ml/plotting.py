import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import random

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)


def plot(x: pl.DataFrame, y: pl.DataFrame, y_hat: pl.DataFrame):

    if bool(len(y_hat.select(pl.col(r'^X_P.*$')).columns)):
        plot_active_power(x=x, y=y, y_hat=y_hat)
    if bool(len(y_hat.select(pl.col(r'^X_Q.*$')).columns)):
        plot_reactive_power(x=x, y=y, y_hat=y_hat)


def plot_active_power(x: pl.DataFrame, y: pl.DataFrame, y_hat: pl.DataFrame):
    x = x.with_columns(agg_kwh=x.select(pl.col(r'^X_P.*$')).sum_horizontal())
    y = y.with_columns(agg_kwh=y.select(pl.col(r'^X_P.*$')).sum_horizontal())
    y_hat = y_hat.with_columns(agg_kwh=y_hat.select(pl.col(r'^X_P.*$')).sum_horizontal())

    fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,10))

    axs[0].plot(x['t_timestamp'], x['agg_kwh'], label='$P^{hist}_{kWh}$', color='#7393B3', linewidth=1)
    axs[0].plot(y['t_timestamp'], y['agg_kwh'], label='$P^{real}_{kWh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
    axs[0].plot(y_hat['t_timestamp'], y_hat['agg_kwh'], label='$P^{pred}_{kWh}$', color='#0000FF', linewidth=1, marker='x')
    axs[0].set_title(f"Aggregated active power {len(y)}h forecast from {y['t_timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    axs[0].set_ylabel('$P^{sum}_{kWh}$')
    axs[0].legend(loc='lower left')

    meter = random.choice(y.select(pl.col(r'^X_P.*$')).columns)

    axs[1].plot(x['t_timestamp'], x[meter], label='$P^{hist}_{kWh}$', color='#7393B3', linewidth=1)
    axs[1].plot(y['t_timestamp'], y[meter], label='$P^{real}_{kWh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
    axs[1].plot(y_hat['t_timestamp'], y_hat[meter], label='$P^{pred}_{kWh}$', color='#0000FF', linewidth=1, marker='x')
    axs[1].set_title(f"Active power {len(y)}h forecast from {y['t_timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} for meter {meter}")
    axs[1].set_ylabel('$P_{kWh}$')
    axs[1].legend(loc='lower left')

    fig.text(0.5, 0.04, 'time', ha='center')
    xticks = pd.date_range(x['t_timestamp'].min(), y['t_timestamp'].max(), (x.shape[0]+y.shape[0])//6)
    xticks_labels = [ f'{date.hour-12}PM' if date.hour > 12 else f'{date.hour}AM' for date in xticks]
    axs[1].xaxis.set_ticks(xticks, xticks_labels)

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()


def plot_reactive_power(x: pl.DataFrame, y: pl.DataFrame, y_hat: pl.DataFrame):
    x = x.with_columns(agg_kwh=x.select(pl.col(r'^X_Q.*$')).sum_horizontal())
    y = y.with_columns(agg_kwh=y.select(pl.col(r'^X_Q.*$')).sum_horizontal())
    y_hat = y_hat.with_columns(agg_kwh=y_hat.select(pl.col(r'^X_P.*$')).sum_horizontal())

    fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,10))

    axs[0].plot(x['t_timestamp'], x['agg_kwh'], label='$Q^{hist}_{kVArh}$', color='#7393B3', linewidth=1)
    axs[0].plot(y['t_timestamp'], y['agg_kwh'], label='$Q^{real}_{kVArh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
    axs[0].plot(y_hat['t_timestamp'], y_hat['agg_kwh'], label='$Q^{pred}_{kVArh}$', color='#0000FF', linewidth=1, marker='x')
    axs[0].set_title(f"Aggregated reactive power {len(y)}h forecast from {y['t_timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    axs[0].set_ylabel('$Q^{sum}_{kWh}$')
    axs[0].legend(loc='lower left')

    meter = random.choice(y.select(pl.col(r'^X_Q.*$')).columns)

    axs[1].plot(x['t_timestamp'], x[meter], label='$Q^{hist}_{kVArh}$', color='#7393B3', linewidth=1)
    axs[1].plot(y['t_timestamp'], y[meter], label='$Q^{real}_{kVArh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
    axs[1].plot(y_hat['t_timestamp'], y_hat[meter], label='$Q^{pred}_{kVArh}$', color='#0000FF', linewidth=1, marker='x')
    axs[1].set_title(f"Active power {len(y)}h forecast from {y['t_timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} for meter {meter}")
    axs[1].set_ylabel('$Q_{kVArh}$')
    axs[1].legend(loc='lower left')

    fig.text(0.5, 0.04, 'time', ha='center')
    xticks = pd.date_range(x['t_timestamp'].min(), y['t_timestamp'].max(), (x.shape[0]+y.shape[0])//6)
    xticks_labels = [ f'{date.hour-12}PM' if date.hour > 12 else f'{date.hour}AM' for date in xticks]
    axs[1].xaxis.set_ticks(xticks, xticks_labels)

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()


def plot_aggregate(x: pl.DataFrame, y: pl.DataFrame, y_hat: pl.DataFrame):
    x = x.with_columns(q_sum_kvarh=x.select(pl.col(r'^X_Q.*$')).sum_horizontal(), p_sum_kwh=x.select(pl.col(r'^X_P.*$')).sum_horizontal())
    y = y.with_columns(q_sum_kvarh=y.select(pl.col(r'^X_Q.*$')).sum_horizontal(), p_sum_kwh=y.select(pl.col(r'^X_P.*$')).sum_horizontal())
    y_hat = y_hat.with_columns(q_sum_kvarh=y_hat.select(pl.col(r'^X_Q.*$')).sum_horizontal(), p_sum_kwh=y_hat.select(pl.col(r'^X_P.*$')).sum_horizontal())

    fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,10))

    axs[0].plot(x['t_timestamp'], x['p_sum_kwh'], label='$P^{hist}_{kWh}$', color='#7393B3', linewidth=1)
    axs[0].plot(y['t_timestamp'], y['p_sum_kwh'], label='$P^{real}_{kWh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
    axs[0].plot(y_hat['t_timestamp'], y_hat['p_sum_kwh'], label='$P^{pred}_{kWh}$', color='#0000FF', linewidth=1, marker='x')
    axs[0].set_title(f"Aggregated active power {len(y)}h forecast from {y['t_timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    axs[0].set_ylabel('$P^{sum}_{kWh}$')
    axs[0].legend(loc='lower left')

    axs[1].plot(x['t_timestamp'], x['q_sum_kvarh'], label='$Q^{hist}_{kVArh}$', color='#7393B3', linewidth=1)
    axs[1].plot(y['t_timestamp'], y['q_sum_kvarh'], label='$Q^{real}_{kVArh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
    axs[1].plot(y_hat['t_timestamp'], y_hat['q_sum_kvarh'], label='$Q^{pred}_{kVArh}$', color='#0000FF', linewidth=1, marker='x')
    axs[1].set_title(f"Aggregated reactive power {len(y)}h forecast from {y['t_timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    axs[1].set_ylabel('$Q^{sum}_{kWh}$')
    axs[1].legend(loc='lower left')

    fig.text(0.5, 0.04, 'time', ha='center')
    xticks = pd.date_range(x['t_timestamp'].min(), y['t_timestamp'].max(), (x.shape[0]+y.shape[0])//6)
    xticks_labels = [ f'{date.hour-12}PM' if date.hour > 12 else f'{date.hour}AM' for date in xticks]
    axs[1].xaxis.set_ticks(xticks, xticks_labels)

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

