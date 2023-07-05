import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class ScoreType:
    CustomScore = "CustomScore"
    Probability = "Probability"
    MegaScore = "MegaScore"
    SubScore = "SubScore"


class MegaScore:
    mapping_base = {
        500: 0.128,
        550: 0.0671,
        600: 0.0341,
        650: 0.017,
        700: 0.0084,
        750: 0.0041,
        800: 0.002,
        850: 0.001,
    }
    score_cap = 1000
    score_floor = 300


class SubScore:
    mapping_base = {
        10: 0.128,
        15: 0.0987,
        20: 0.0755,
        25: 0.0574,
        30: 0.0434,
        35: 0.0327,
        40: 0.0246,
        45: 0.0185,
        50: 0.0138,
        55: 0.0104,
        60: 0.0077,
        65: 0.0058,
        70: 0.0043,
        75: 0.0032,
        80: 0.0024,
        85: 0.0018,
        90: 0.0013,
        95: 0.001,
    }
    score_cap = 100
    score_floor = 0


class Score(TransformerMixin):
    def __init__(self, **kwargs):
        self.n_bins = kwargs.get("n_bins", 25)
        self.n_degree = kwargs.get("n_degree", 1)
        self.score_type = kwargs.get("score_type", None)

        self.mapping_base = None
        self.score_cap = None
        self.score_floor = None

        self.summary = None
        self.coef = None
        self.mapping_intercept = None
        self.mapping_slope = None

    def fit(self, sr_prob, sr_label, **kwargs):
        self.score_type = kwargs.get("score_type", ScoreType.CustomScore)

        mapping_base = kwargs.get("mapping_base", MegaScore.mapping_base)
        score_cap = kwargs.get("score_cap", MegaScore.score_cap)
        score_floor = kwargs.get("score_floor", MegaScore.score_floor)

        if self.score_type == ScoreType.CustomScore:
            self.mapping_base = mapping_base
            self.score_cap, self.score_floor = score_cap, score_floor
            self.mapping_slope, self.mapping_intercept = self.get_params(self.mapping_base)
        elif self.score_type == ScoreType.MegaScore:
            self.mapping_base = MegaScore.mapping_base
            self.score_cap, self.score_floor = (
                MegaScore.score_cap,
                MegaScore.score_floor,
            )
            self.mapping_slope, self.mapping_intercept = self.get_params(self.mapping_base)
        elif self.score_type == ScoreType.SubScore:
            self.mapping_base = SubScore.mapping_base
            self.score_cap, self.score_floor = SubScore.score_cap, SubScore.score_floor
            self.mapping_slope, self.mapping_intercept = self.get_params(self.mapping_base)
        else:
            pass

        df_data = pd.DataFrame({"yprob": list(sr_prob), "label": list(sr_label)})

        # step 1: convert prob to lnodd
        df_data["lnodds_prob"] = df_data["yprob"].apply(lambda x: self.prob2lnodds(x))
        # step 2: cut the lnodd into bins
        df_data["lnodds_prob_bin"] = pd.qcut(df_data["lnodds_prob"], self.n_bins, duplicates="drop")
        # step 3: stat user count, bad rate in each bin
        df_bin = df_data.groupby("lnodds_prob_bin").agg(
            total_user=("label", "count"),
            bad_rate=("label", "mean"),
            lnodds_prob_mean_x=("lnodds_prob", "mean"),
        )
        # step 4: smooth the bad rate if there is 0
        df_bin["adj_bad_rate"] = df_bin.apply(lambda x: max(x["bad_rate"], 1 / x["total_user"], 0.0001), axis=1)

        # step 5: convert bad rate to lnodd
        df_bin["lnodds_bad_rate_y"] = df_bin["adj_bad_rate"].apply(lambda x: self.prob2lnodds(x))

        # step 6: learn the coefficient: lnodds_bad_rate_y = f(lnodds_prob)
        self.coef = np.polyfit(df_bin["lnodds_prob_mean_x"], df_bin["lnodds_bad_rate_y"], self.n_degree)

        # record summary
        self.summary = df_bin[
            [
                "total_user",
                "bad_rate",
                "adj_bad_rate",
                "lnodds_prob_mean_x",
                "lnodds_bad_rate_y",
            ]
        ]

        return self

    def transform(self, sr_prob):
        lst_lnodds_prob = [self.prob2lnodds(x) for x in list(sr_prob)]
        lst_lnodds_cal_prob = [np.poly1d(self.coef)(x) for x in lst_lnodds_prob]

        if self.score_type == ScoreType.Probability:
            lst_cal_prob = [self.lnodds2prob(x) for x in lst_lnodds_cal_prob]
            return lst_cal_prob
        else:
            lst_score = [self.mapping_intercept + self.mapping_slope * x for x in lst_lnodds_cal_prob]
            lst_score = [max(x, self.score_floor) for x in lst_score]
            lst_score = [min(x, self.score_cap) for x in lst_score]
            return lst_score

    @classmethod
    def prob2lnodds(cls, prob):
        if prob == 0:
            lnodds = np.log(np.finfo(float).eps)
        elif prob == 1:
            lnodds = np.log(prob / (1 - prob + np.finfo(float).eps))
        else:
            lnodds = np.log(prob / (1 - prob))
        return lnodds

    @classmethod
    def lnodds2prob(cls, lnodds):
        prob = 1 - 1 / (np.exp(lnodds) + 1)
        return prob

    @classmethod
    def get_params(cls, dict_base):
        lst_score = sorted(dict_base.keys())
        lst_bad_rate = sorted(dict_base.values(), reverse=True)
        lst_lnodds_bad_rate = [cls.prob2lnodds(x) for x in lst_bad_rate]

        score_min, score_max = lst_score[0], lst_score[-1]
        lnodds_min, lnodds_max = lst_lnodds_bad_rate[0], lst_lnodds_bad_rate[-1]

        slope = (score_max - score_min) / (lnodds_max - lnodds_min)
        intercept = score_max - slope * lnodds_max
        return slope, intercept
