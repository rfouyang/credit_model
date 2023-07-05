import numpy as np
import pandas as pd
from util import metric_helper


class FTReport:
    @classmethod
    def get_report(cls, df_bin, features, label, **kwargs):
        exclude = kwargs.get("exclude", list())

        lst_df = list()
        for c in features:
            if c in exclude + [label]:
                continue

            df_stat = (
                df_bin[[c, label]]
                .groupby(c)
                .agg(
                    total=(f"{label}", "count"),
                    bad=(f"{label}", "sum"),
                    bad_rate=(f"{label}", "mean"),
                )
            )
            df_stat = df_stat.reset_index().rename(columns={c: "bin"})
            df_stat["feature"] = c
            df_stat["total_pct"] = df_stat["total"] / df_stat["total"].sum()

            df_stat = df_stat[["feature", "bin", "total_pct", "total", "bad", "bad_rate"]]

            sr = pd.Series(
                [None, None, None, None, None, None],
                index=["feature", "bin", "total_pct", "total", "bad", "bad_rate"],
            )
            df_stat = pd.concat([df_stat, pd.DataFrame([sr])], ignore_index=True)
            #df_stat = df_stat.append(sr, ignore_index=True)
            lst_df.append(df_stat)

        df_report = pd.concat(lst_df)

        return df_report

    @classmethod
    def __get_stat(cls, df_woe, c, label):
        df_stat = df_woe.groupby(c).agg(
            total=(label, 'count'),
            good=(label, lambda x: (1 - x).sum()),
            bad=(label, 'sum'),
            bad_rate=(label, 'mean')).reset_index()

        df_stat['good_density'] = df_stat['good'] / df_stat['good'].sum()
        df_stat['bad_density'] = df_stat['bad'] / df_stat['bad'].sum()
        df_stat['feature'] = c
        df_stat = df_stat.rename(columns={c: 'woe'})
        df_stat = df_stat[['feature', 'total', 'good', 'bad', 'good_density', 'bad_density', 'woe']]

        df_stat['woe_temp'] = np.log(df_stat['good_density'] / df_stat['bad_density'])
        df_stat['iv'] = (df_stat['good_density'] - df_stat['bad_density']) * df_stat['woe_temp']

        return df_stat

    @classmethod
    def eval_metrics(cls, df_woe, features, label):
        lst_item = list()
        for c in features:
            df_sub = cls.__get_stat(df_woe, c, label)
            iv = df_sub['iv'].sum()
            auc = metric_helper.Metric.get_auc(df_woe[label], df_woe[c])
            ks = metric_helper.Metric.get_ks(df_woe[label], df_woe[c])
            item = {
                'feature': c,
                'iv': iv,
                'auc': auc,
                'ks': ks
            }
            lst_item.append(item)
        df_eval = pd.DataFrame(lst_item)
        return df_eval


class ScoreReport:
    @classmethod
    def get_report(cls, df_score, score_bin, label):
        df_stat = df_score.groupby(score_bin).agg(total=(label, "count"), bad=(label, "sum"), bad_rate=(label, "mean"))

        df_stat = df_stat.reset_index()
        df_stat["total_pct"] = df_stat["total"] / df_stat["total"].sum()
        df_report = df_stat[[score_bin, "total_pct", "total", "bad", "bad_rate"]]

        return df_report


class ModelReport:
    @classmethod
    def get_report(cls, df_score, sample_type, score, label):
        sr_auc = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_auc(r[label], r[score]))
        sr_ks = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_ks(r[label], r[score]))
        sr_gini = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_gini(r[label], r[score]))

        sr_auc.name = "auc"
        sr_ks.name = "ks"
        sr_gini.name = "gini"

        df_report = pd.DataFrame([sr_auc, sr_ks, sr_gini])
        return df_report

    @classmethod
    def get_lr_report(cls, df_score, sample_type, score, label):
        sr_rmse = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_rmse(r[label], r[score]))
        sr_mse = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_mse(r[label], r[score]))
        sr_mae = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_mae(r[label], r[score]))
        sr_r2 = df_score.groupby(sample_type).apply(lambda r: metric_helper.Metric.get_r2(r[label], r[score]))

        sr_rmse.name = "rmse"
        sr_mse.name = "mse"
        sr_mae.name = "mae"
        df_report = pd.DataFrame([sr_rmse, sr_mse, sr_mae])
        return df_report
