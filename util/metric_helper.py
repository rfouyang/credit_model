import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)


class Metric:
    @classmethod
    def get_auc(cls, ytrue, yprob):
        if sum(ytrue) == 0 or sum(ytrue) == len(ytrue):
            return None

        auc = roc_auc_score(ytrue, yprob)
        if auc < 0.5:
            auc = 1 - auc
        return auc

    @classmethod
    def get_ks(cls, ytrue, yprob):
        if sum(ytrue) == 0:
            return None

        fpr, tpr, thr = roc_curve(ytrue, yprob)
        ks = max(abs(tpr - fpr))
        return ks

    @classmethod
    def get_gini(cls, ytrue, yprob, **kwargs):
        auc = cls.get_auc(ytrue, yprob)
        gini = 2 * auc - 1

        return gini

    @classmethod
    def get_cm(cls, ytrue, yprob, thr):
        ylabel = np.where(yprob > thr, 1, 0)
        cm = confusion_matrix(ytrue, ylabel)
        return cm

    @classmethod
    def get_rmse(cls, ytrue, yprob):
        mse = mean_squared_error(ytrue, yprob)
        return np.sqrt(mse)

    @classmethod
    def get_mse(cls, ytrue, yprob):
        return mean_squared_error(ytrue, yprob)

    @classmethod
    def get_mae(cls, ytrue, yprob):
        return mean_absolute_error(ytrue, yprob)

    @classmethod
    def get_r2(cls, ytrue, yprob):
        return r2_score(ytrue, yprob)

    @classmethod
    def get_stat(cls, sr_feature, sr_label):
        var = sr_feature.name
        df_data = pd.DataFrame({"val": sr_feature, "label": sr_label})

        # statistics of total count, total ratio, bad count, bad rate
        df_stat = df_data.groupby("val").agg(total=("label", "count"), bad=("label", "sum"), bad_rate=("label", "mean"))
        df_stat["var"] = var
        df_stat["good"] = df_stat["total"] - df_stat["bad"]
        df_stat["total_ratio"] = df_stat["total"] / df_stat["total"].sum()
        df_stat["good_density"] = df_stat["good"] / df_stat["good"].sum()
        df_stat["bad_density"] = df_stat["bad"] / df_stat["bad"].sum()

        eps = np.finfo(np.float32).eps
        df_stat.loc[:, "iv"] = (df_stat["bad_density"] - df_stat["good_density"]) * np.log(
            (df_stat["bad_density"] + eps) / (df_stat["good_density"] + eps)
        )

        cols = ["var", "total", "total_ratio", "bad", "bad_rate", "iv", "val"]
        df_stat = df_stat.reset_index()[cols].set_index("var")
        return df_stat

    @classmethod
    def get_iv(cls, sr_feature, sr_label):
        df_stat = cls.get_stat(sr_feature, sr_label)
        return df_stat["iv"].sum()
