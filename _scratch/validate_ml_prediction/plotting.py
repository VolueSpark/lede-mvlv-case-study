from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import polars as pl
import os

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

import matplotlib
matplotlib.rc('font', **font)

#
# *_in -> from energy end consumer producing and exporting to grid
# *_out -> from grid energy imported by energy end consumer

def plot(from_date: datetime, days: int=1, path=''):

    folder = f"{from_date}_{from_date+timedelta(days=days)}"
    folder_path = os.path.join(path, folder)

    p = pl.read_parquet(os.path.join(folder_path, 'predictions'))
    m = pl.read_parquet(os.path.join(folder_path, 'measurements'))


    fig, axs = plt.subplots(2,1)
    fig.suptitle(f"Aggregated net active and reactive load profiles for Lede")

    axs[0].plot(m['datetime'], m['sum_p_kwh_out']-m['sum_p_kwh_in'], label='$P^{meas}_{aggr.}$', color='#57b9ff', linestyle=(0, (5, 1)),linewidth=0.7)
    axs[0].plot(p['datetime'], p['sum_p_kwh_out']-p['sum_p_kwh_in'], label='$P^{pred}_{aggr.}$', color='#add8e6', linewidth=0.7)
    axs[0].legend(loc='upper left')
    axs[0].set_xticklabels(labels=axs[0].get_xticklabels(), minor=False, rotation=15)
    axs[0].tick_params(axis ='y', labelcolor ='#57b9ff')
    axs[0].set_ylabel('kWh', color='#57b9ff')
    axs[0].set_title('Net Active power')

    axs[1].plot(m['datetime'], m['sum_q_kvarh_out']-m['sum_q_kvarh_in'], label='$P^{meas}_{aggr}$', color='#5CE65C', linestyle=(0, (5, 1)), linewidth=0.7)
    axs[1].plot(p['datetime'], p['sum_q_kvarh_out']-p['sum_q_kvarh_in'], label='$P^{pred}_{aggr}$', color='#008000', linewidth=0.7)
    axs[1].legend(loc='upper left')
    axs[1].set_xticklabels(labels=axs[1].get_xticklabels(), minor=False, rotation=15)
    axs[1].tick_params(axis ='y', labelcolor ='#008000')
    axs[1].set_ylabel('kVArh', color='#008000')
    axs[1].set_title('Net Reactive power')

    plt.tight_layout()

    fig.savefig(os.path.join(folder_path, "aggregated_validation.png"), dpi=200)

def plot_processed_meter(original: pl.DataFrame, processed: pl.DataFrame, feature: str):
    o = original.sort(by='datetime', descending=False)
    p = processed.sort(by='datetime', descending=False)

    meter_id = o['meter_id'].unique().item()
    fig, ax = plt.subplots(1,1)
    fig.suptitle(f"Processed meter data for meter {meter_id}")

    ax.plot(o['datetime'], o[feature], label='$P^{orig}$', color='blue', linewidth=0.7)
    ax.plot(p['datetime'], p[feature], label='$P^{pros}$', color='red', linewidth=0.7, linestyle='--')
    ax.legend(loc='upper left')
    ax.set_xticklabels(labels=ax.get_xticklabels(), minor=False, rotation=15)
    ax.tick_params(axis ='y', labelcolor ='#57b9ff')
    ax.set_ylabel('Production [kWh]', color='#57b9ff')
    ax.set_title(feature)

    '''
    axs[0].plot(o['datetime'], -o['p_kwh_in'], label='$P^{orig}$', color='#57b9ff', linestyle=(0, (5, 1)),linewidth=0.7)
    axs[0].plot(p['datetime'], -p['p_kwh_in'], label='$P^{pros}$', color='#add8e6', linewidth=0.7)
    axs[0].legend(loc='upper left')
    axs[0].set_xticklabels(labels=axs[0].get_xticklabels(), minor=False, rotation=15)
    axs[0].tick_params(axis ='y', labelcolor ='#57b9ff')
    axs[0].set_ylabel('Production [kWh]', color='#57b9ff')
    axs[0].set_title('Active power')

    ax0 = axs[0].twinx()
    ax0.plot(o['datetime'], o['p_kwh_out'], label='$P^{orig}$', color='#5CE65C', linestyle=(0, (5, 1)), linewidth=0.7)
    ax0.plot(p['datetime'], p['p_kwh_out'], label='$P^{pros}$', color='#008000', linewidth=0.7)
    ax0.tick_params(axis ='y', labelcolor ='#008000')
    ax0.set_ylabel('Consumption [kWh]', color='#008000')
    ax0.legend(loc='upper right')

    axs[1].plot(o['datetime'], -o['q_kvarh_in'], label='$Q^{orig}$', color='#57b9ff', linestyle=(0, (5, 1)), linewidth=0.7)
    axs[1].plot(p['datetime'], -p['q_kvarh_in'], label='$Q^{pros}$', color='#add8e6', linewidth=0.7)
    axs[1].legend(loc='upper left')
    axs[1].set_xticklabels(labels=axs[0].get_xticklabels(), minor=False, rotation=15)
    axs[1].tick_params(axis ='y', labelcolor ='#57b9ff')
    axs[1].set_ylabel('Production [kVArh]', color='#57b9ff')
    axs[1].set_title('Reactive power')

    ax1 = axs[1].twinx()
    ax1.plot(o['datetime'], o['q_kvarh_out'], label='$Q^{orig}$', color='#5CE65C', linestyle=(0, (5, 1)), linewidth=0.7)
    ax1.plot(p['datetime'], p['q_kvarh_out'], label='$Q^{pros}$', color='#008000', linewidth=0.7)
    ax1.tick_params(axis ='y', labelcolor ='#008000')
    ax1.set_ylabel('Consumption [kVArh]', color='#008000')
    ax1.legend(loc='upper right')
    '''

    plt.tight_layout()
    plt.show()

