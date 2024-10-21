from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import scipy, os

PATH = os.path.dirname(__file__)

def annotate_metrics(ax, metrics: dict):
    ax.annotate(
        '\n'.join([f'{key}={value}' for key, value in metrics.items()]),
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0)
    )


def decorator_confidence(func):

    def confidence_interval(data:pl.DataFrame, confidence: float) ->pl.DataFrame:
        ci_analysis = []
        def ci(data:pl.Series, confidence):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a, axis=0), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

            return m, m-h, m+h

        prediction_range = range(data.filter(pl.col('type')=='target')['offset'].min(), data.filter(pl.col('type')=='target')['offset'].max()+1)
        for power_type in ['P', 'Q']:
            wildcard = f'^X_{power_type}.*$'
            scaler = MinMaxScaler()
            scaler.fit(data.filter(pl.col('type')=='target').select(wildcard))
            scale_coef = sum([ max(val, abs(scaler.data_min_[i])) for i, val in enumerate(scaler.data_max_)])
            for k in prediction_range:
                pred_k = data.filter((pl.col('offset') == k) & (pl.col('type') == 'forecast')).select(wildcard)
                real_k = data.filter((pl.col('offset') == k) & (pl.col('type') == 'target')).select(wildcard)

                # absolute sum % error on maximum sum target value
                error_k = abs((pred_k - real_k).to_numpy()).sum(axis=1)/scale_coef*100 # absolute error [power unit/meter]
                mean_k, ci_lb_k, ci_ub_k = ci(data=error_k, confidence=confidence)
                ci_analysis.append(
                    {
                        'type': power_type,
                        'k': k - prediction_range[0],
                        'mean': mean_k,
                        'ci_lb': ci_lb_k,
                        'ci_ub': ci_ub_k}
                )
        return pl.from_dicts(ci_analysis)

    def inner(cls, *args, **kwargs):

        os.makedirs(os.path.join(cls.path, 'logs/analysis'), exist_ok=True)
        data = pl.read_parquet(os.path.join(cls.path, 'data/gold/data.parquet'))

        confidence = 0.95
        ci_analysis = confidence_interval(data=data, confidence=confidence)

        n = ci_analysis.n_unique('type')
        fig, axs = plt.subplots(n,1, sharex=True, figsize=(15,10))

        for i, (type, data) in enumerate(ci_analysis.group_by('type')):
            axs[i].plot(data['k'], data['mean'], color='#58855c', linewidth=1, linestyle='--')
            axs[i].plot(data['k'], data['ci_lb'], color='#58855c', linewidth=1)
            axs[i].plot(data['k'], data['ci_ub'], color='#58855c', linewidth=1)
            axs[i].fill_between(data['k'], y1=data['ci_lb'], y2=data['ci_ub'], alpha=0.2,color='#58855c')

            axs[i].set_ylabel('$\\frac{\sum{|{Err}_i|}}{\sum{|y_i^{max}|}} \\times{100}$')
            axs[i].set_xlabel('Forecast timestep [t+k]')
            axs[i].set_title("Aggregated error confidence for {0}".format(type[0]))

            annotate_metrics(axs[i], metrics={'CI':confidence })


        fig.tight_layout()
        fig.savefig(os.path.join(cls.path, 'logs/analysis/confidence.png'), dpi=300)

    return inner

