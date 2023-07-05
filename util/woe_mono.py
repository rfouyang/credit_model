import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from tqdm import tqdm

warnings.simplefilter(action="ignore")


class WOEMono(TransformerMixin):
    def __init__(self, **kwargs):
        self.nbins = kwargs.get("nbins", 5)
        self.min_bin_rate = kwargs.get("min_bin_rate", 0.02)
        self.min_bin_size = kwargs.get("min_bin_size", 50)
        self.min_bin_adjusted_rate = self.min_bin_rate
        self.min_missing_bad_cnt = kwargs.get("min_missing_bad_cnt", 30)

        self.feature_list = None
        self.categorical_features = None
        self.missing_values = None
        self.missing_logic = None
        self.bin_info = None

    def fit_numerical(self, df, c):
        df_missing, df_data = self.split_data_by_missing(df, c)

        stat_missing = self.stat_group(df_missing, c, "label")
        df_data.loc[:, c] = df_data.loc[:, c].apply(lambda x: round(x, 4))
        stat_data = self.stat_group(df_data, c, "label")

        df_bin = self.stat_bin_info(stat_missing, stat_data, c)

        return df_bin

    def fit_categorical(self, df, c):
        df_missing, df_data = self.split_data_by_missing(df, c)

        stat_missing = self.stat_group(df_missing, c, "label")
        stat_data = self.stat_group(df_data, c, "label")

        df_bin = self.stat_bin_info(stat_missing, stat_data, c)

        return df_bin

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        self.missing_values = kwargs.get("missing_values", list())
        self.categorical_features = kwargs.get("categorical_features", list())
        self.feature_list = self.get_feature_list(df_xtrain, **kwargs)
        missing_logic = kwargs.get("missing_logic", dict())

        df_train = df_xtrain.copy(deep=True)
        df_train["label"] = df_ytrain
        self.min_bin_adjusted_rate = max(self.min_bin_rate, self.min_bin_size / float(len(df_train)))

        lst_bin = list()

        self.missing_logic = dict()
        for c in self.feature_list:
            self.missing_logic[c] = "high_risk"
        for k, v in missing_logic.items():
            self.missing_logic[k] = v

        for c in tqdm(self.feature_list):
            start = time.time()
            if c in self.categorical_features:
                df_bin = self.fit_categorical(df_train, c)
            else:
                df_bin = self.fit_numerical(df_train, c)
            finish = time.time()
            df_bin["time"] = finish - start
            lst_bin.append(df_bin)

        self.bin_info = pd.concat(lst_bin)
        self.bin_info = self.__adjust_woe(self.bin_info)

        return self

    def transform(self, df_xtest, **kwargs):
        method = kwargs.get("method", "woe")

        df_trans = df_xtest.copy(deep=True)
        df_trans_res = df_xtest.copy(deep=True)
        feature_list = self.get_feature_list(df_trans, **kwargs)
        feature_list = sorted(set(feature_list) & set(self.feature_list))

        for c in tqdm(feature_list):
            if c in self.categorical_features:
                df_missing = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "cat_missing")]
                df_data = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "cat_normal")]

                if method == "woe":
                    if self.missing_logic[c] == "high_risk":
                        df_trans_res.loc[:, c] = df_data["woe_min"].values[0]
                    elif self.missing_logic[c] == "low_risk":
                        df_trans_res.loc[:, c] = df_data["woe_max"].values[0]
                    else:
                        df_trans_res.loc[:, c] = 0

                df_trans.loc[:, c] = df_trans.loc[:, c].fillna("NA")
                for idx, row in df_data.iterrows():
                    df_trans_res.loc[df_trans[c] == row["min"], c] = row[method]
                for idx, row in df_missing.iterrows():
                    df_trans_res.loc[df_trans[c] == row["min"], c] = row[method]
            else:
                df_missing = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "num_missing")]
                df_data = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "num_normal")]

                if method == "woe":
                    if self.missing_logic[c] == "high_risk":
                        df_trans_res.loc[:, c] = df_data["woe_min"].values[0]
                    elif self.missing_logic[c] == "low_risk":
                        df_trans_res.loc[:, c] = df_data["woe_max"].values[0]
                    else:
                        df_trans_res.loc[:, c] = 0

                df_trans.loc[:, c] = df_trans.loc[:, c].fillna(-990000)
                df_trans.loc[:, c] = df_trans.loc[:, c].apply(lambda x: round(x, 4))

                for idx, row in df_data.iterrows():
                    df_trans_res.loc[
                        (float(row["min"]) < df_trans[c]) & (df_trans[c] <= float(row["max"])),
                        c,
                    ] = row[method]

                for idx, row in df_missing.iterrows():
                    df_trans_res.loc[df_trans[c] == float(row["min"]), c] = row[method]

        return df_trans_res

    @classmethod
    def get_feature_list(cls, df, **kwargs):
        exclude = kwargs.get("exclude", list())
        feature_list = kwargs.get("feature_list", df.columns.tolist())
        feature_list = sorted(set(feature_list) & set(df.columns.tolist()) - set(exclude))
        return feature_list

    def split_data_by_missing(self, df, c):
        df_missing = df.loc[df[c].apply(lambda x: self.is_missing(x))]
        df_data = df.loc[df[c].apply(lambda x: not self.is_missing(x))]

        if c in self.categorical_features:
            df_missing.loc[:, c] = df_missing.loc[:, c].fillna("NA")
        else:
            df_missing.loc[:, c] = df_missing.loc[:, c].fillna(-990000)

        return df_missing, df_data

    def is_missing(self, x):
        if pd.isna(x) or x in self.missing_values:
            return True
        else:
            return False

    def stat_group(self, df, c, label):
        stat = df[label].groupby([df[c], df[label]]).count().unstack().reset_index().fillna(0)

        stat = stat.sort_values(by=c, ascending=True)
        stat[c] = stat[c].astype(str)
        stat = stat.rename(columns={0: "good", 1: "bad", c: "bin"})
        stat = stat.reset_index(drop=True)
        stat.columns.name = None

        if "good" not in stat.columns:
            stat["good"] = 0
        if "bad" not in stat.columns:
            stat["bad"] = 0
        stat["var"] = c

        return stat

    def stat_bin_info(self, stat_missing, stat_data, c):
        good_data = stat_data["good"].sum()
        bad_data = stat_data["bad"].sum()

        if c in self.categorical_features:
            if len(stat_missing) > 0:
                good_missing = stat_missing["good"].sum()
                bad_missing = stat_missing["bad"].sum()

                bin_missing = self.__stat_bin(
                    stat_missing,
                    stat_missing.index.tolist(),
                    "cat_missing",
                    good_data,
                    bad_data,
                )
            else:
                good_missing = 0
                bad_missing = 0

                bin_missing = pd.DataFrame(
                    [
                        {
                            "var": c,
                            "type": "cat_missing",
                            "bin": "NA",
                            "min": "NA",
                            "max": "NA",
                            "woe": -np.inf,
                            "iv": 0,
                            "total": 0,
                            "ratio": 0,
                            "bad": 0,
                            "bad_rate": 0,
                        }
                    ]
                )

            bin_data = self.__stat_bin(
                stat_data,
                stat_data.index.tolist(),
                "cat_normal",
                good_missing,
                bad_missing,
            )
        else:
            if len(stat_missing) > 0:
                good_missing = stat_missing["good"].sum()
                bad_missing = stat_missing["bad"].sum()

                bin_missing = self.__stat_bin(
                    stat_missing,
                    stat_missing.index.tolist(),
                    "num_missing",
                    good_data,
                    bad_data,
                )
            else:
                good_missing = 0
                bad_missing = 0

                bin_missing = pd.DataFrame(
                    [
                        {
                            "var": c,
                            "type": "num_missing",
                            "bin": -990000,
                            "min": -990000,
                            "max": -990000,
                            "woe": -np.inf,
                            "iv": 0,
                            "total": 0,
                            "ratio": 0,
                            "bad": 0,
                            "bad_rate": 0,
                        }
                    ]
                )

            total = good_data + bad_data + good_missing + bad_missing
            knots = self.init_knots(stat_data, total, self.nbins)
            knots = self.combine_bins(stat_data, self.nbins, knots)
            bin_data = self.__stat_bin(stat_data, knots, "num_normal", good_missing, bad_missing)

        bin_total = pd.concat([bin_data, bin_missing], axis=0).reset_index(drop=True)

        woe_max = bin_data["woe"].max()
        woe_min = bin_data["woe"].min()

        bin_total["bin"] = bin_total["bin"].apply(lambda x: str(x))
        for idx, row in bin_total.iterrows():
            if row["type"] == "num_normal":
                v = f"{idx:02}.{row['bin']}"
            else:
                v = f"{idx:02}.{{{row['bin']}}}"

            bin_total.at[idx, "bin"] = v

        bin_total["woe_max"] = woe_max
        bin_total["woe_min"] = woe_min

        return bin_total

    def stat_bin_info_update_bin(self, stat_missing, stat_data):

        if len(stat_missing) > 0:
            good_missing = stat_missing["good"].sum()
            bad_missing = stat_missing["bad"].sum()
            good_data = stat_data["good"].sum()
            bad_data = stat_data["bad"].sum()

            bin_missing = self.__stat_bin(
                stat_missing,
                stat_missing.index.tolist(),
                "num_missing",
                good_data,
                bad_data,
            )
        else:
            good_missing = stat_missing["good"].sum()
            bad_missing = stat_missing["bad"].sum()
            bin_missing = pd.DataFrame()

        bin_data = self.__stat_bin_update(stat_data, "num_normal", good_missing, bad_missing)

        bin_total = pd.concat([bin_data, bin_missing], axis=0).reset_index(drop=True)

        bin_total["bin"] = bin_total["bin"].apply(lambda x: str(x))
        for idx, row in bin_total.iterrows():
            if row["type"] == "num_normal":
                v = f"{idx:02}.{row['bin']}"
            else:
                v = f"{idx:02}.{{{row['bin']}}}"

            bin_total.at[idx, "bin"] = v
        return bin_total

    @classmethod
    def __stat_bin_list(cls, var, bin_type, lst_df, lst_bin, lst_min, lst_max, total_good, total_bad):
        ratio_good = pd.Series(list(map(lambda x: float(sum(x["good"]) + 0.5) / (total_good + 0.5), lst_df)))
        ratio_bad = pd.Series(list(map(lambda x: float(sum(x["bad"]) + 0.5) / (total_bad + 0.5), lst_df)))
        lst_total = list(map(lambda x: sum(x["good"]) + sum(x["bad"]), lst_df))
        lst_ratio = list(lst_total / (total_good + total_bad))
        lst_bad = list(pd.Series(list(map(lambda x: float(sum(x["bad"])), lst_df))))
        lst_rate = list(
            map(
                lambda x: float(sum(x["bad"])) / (sum(x["good"]) + sum(x["bad"])),
                lst_df,
            )
        )
        lst_woe = list(np.log(ratio_good / ratio_bad))
        lst_iv = list((ratio_good - ratio_bad) * np.log(ratio_good / ratio_bad))

        if bin_type == "cat_missing":
            if "NA" not in lst_bin:
                print(f"add edge case: None in {var}")
                lst_bin.append("NA")
                lst_min.append("NA")
                lst_max.append("NA")
                lst_woe.append(0)
                lst_iv.append(0)
                lst_total.append(0)
                lst_ratio.append(0)
                lst_bad.append(0)
                lst_rate.append(0)
        elif bin_type == "num_missing":
            if "-990000.0" not in lst_bin and "-990000" not in lst_bin and -990000 not in lst_bin:
                print(f"add edge case: None in {var}")
                lst_bin.append(-990000)
                lst_min.append(-990000)
                lst_max.append(-990000)
                lst_woe.append(-np.inf)
                lst_iv.append(0)
                lst_total.append(0)
                lst_ratio.append(0)
                lst_bad.append(0)
                lst_rate.append(0)

        df_bin = pd.DataFrame(
            {
                "var": var,
                "type": bin_type,
                "bin": lst_bin,
                "min": lst_min,
                "max": lst_max,
                "woe": lst_woe,
                "iv": lst_iv,
                "total": lst_total,
                "ratio": lst_ratio,
                "bad": lst_bad,
                "bad_rate": lst_rate,
            }
        )
        return df_bin

    @classmethod
    def __stat_bin(cls, stat, knots, bin_type, good, bad):
        var = stat["var"][0]
        total_good = stat["good"].sum() + good
        total_bad = stat["bad"].sum() + bad

        lst_df, lst_bin, lst_min, lst_max = list(), list(), list(), list()
        if bin_type in ["cat_normal", "cat_missing", "num_missing"]:
            for i in knots:
                lst_df.append(stat.loc[i:i])
                lst_bin.append(stat.loc[i]["bin"])
                lst_min.append(stat.loc[i]["bin"])
                lst_max.append(stat.loc[i]["bin"])
        else:
            if len(knots) == 2:
                lst_df.append(stat.loc[knots[0] : knots[1]])
                lst_bin.append(pd.Interval(left=-np.inf, right=np.inf))
                lst_min.append(-np.inf)
                lst_max.append(np.inf)
            else:
                for i in range(1, len(knots)):
                    if i == 1:
                        lst_df.append(stat.loc[knots[i - 1] : knots[i]])
                        val_right = float(stat["bin"][knots[i]])
                        lst_bin.append(pd.Interval(left=-np.inf, right=val_right))
                        lst_min.append(-np.inf)
                        lst_max.append(val_right)
                    else:
                        lst_df.append(stat.loc[knots[i - 1] + 1 : knots[i]])
                        if i == len(knots) - 1:
                            val_left = float(stat["bin"][knots[i - 1]])
                            lst_bin.append(pd.Interval(left=val_left, right=np.inf))
                            lst_min.append(val_left)
                            lst_max.append(np.inf)
                        else:
                            val_left = float(stat["bin"][knots[i - 1]])
                            val_right = float(stat["bin"][knots[i]])
                            lst_bin.append(pd.Interval(left=val_left, right=val_right))
                            lst_min.append(val_left)
                            lst_max.append(val_right)

        df_bin = cls.__stat_bin_list(var, bin_type, lst_df, lst_bin, lst_min, lst_max, total_good, total_bad)

        return df_bin

    @classmethod
    def __stat_bin_update(cls, stat, bin_type, good, bad):
        var = stat["var"][0]
        total_good = stat["good"].sum() + good
        total_bad = stat["bad"].sum() + bad

        lst_df, lst_bin, lst_min, lst_max = list(), list(), list(), list()

        for i in range(len(stat)):
            lst_df.append(stat.loc[i:i])
            lst_bin.append(stat.loc[i, "bin"])
            lst_min.append(stat.loc[i, "bin"].left)
            lst_max.append(stat.loc[i, "bin"].right)

        df_bin = cls.__stat_bin_list(var, bin_type, lst_df, lst_bin, lst_min, lst_max, total_good, total_bad)
        return df_bin

    @classmethod
    def __find_index(cls, lst_val, target):
        return list(filter(lambda v: lst_val[v] == target, range(0, len(lst_val))))

    def get_best_knot_by_ks(self, stat_bin, total, left, right):
        stat = stat_bin.loc[left:right]
        total_cur = stat["good"].sum() + stat["bad"].sum()

        left_add = sum(np.cumsum(stat["good"] + stat["bad"]) < self.min_bin_adjusted_rate * total)
        right_add = sum(np.cumsum(stat["good"] + stat["bad"]) <= total_cur - self.min_bin_adjusted_rate * total)

        left_adjust = left + left_add
        right_adjust = left + right_add - 1

        if right_adjust >= left_adjust:
            if stat["bad"].sum() != 0 and stat["good"].sum() != 0:
                cdf_bad = np.cumsum(stat["bad"]) / stat["bad"].sum()
                cdf_good = np.cumsum(stat["good"]) / stat["good"].sum()

                ks = max(abs(cdf_bad - cdf_good).loc[left_adjust:right_adjust])
                idx = self.__find_index(list(abs(cdf_bad - cdf_good)), ks)
                return stat.index[max(idx)]

            else:
                return None
        else:
            return None

    def get_best_knots_helper(self, stat_bin, total, max_iter, left, right, cur_iter):

        stat = stat_bin.loc[left:right]
        total_cur = stat["good"].sum() + stat["bad"].sum()

        if total_cur < self.min_bin_adjusted_rate * total * 2 or cur_iter >= max_iter:
            return []

        best_knot = self.get_best_knot_by_ks(stat_bin, total, left, right)

        if best_knot is not None:
            left_knots = self.get_best_knots_helper(stat_bin, total, max_iter, left, best_knot, cur_iter + 1)
            right_knots = self.get_best_knots_helper(stat_bin, total, max_iter, best_knot + 1, right, cur_iter + 1)
        else:
            left_knots = []
            right_knots = []
        return left_knots + [best_knot] + right_knots

    def init_knots(self, stat_bin, total, max_iter):
        knots = self.get_best_knots_helper(stat_bin, total, max_iter, 0, len(stat_bin), 0)
        knots = list(filter(lambda x: x is not None, knots))
        knots.sort()

        return knots

    @classmethod
    def eval_iv_mono(cls, stat, knots):
        lst_df = []
        for i in range(1, len(knots)):
            if i == 1:
                lst_df.append(stat.loc[knots[i - 1] : knots[i]])
            else:
                lst_df.append(stat.loc[knots[i - 1] + 1 : knots[i]])
        total_good = stat["good"].sum()
        total_bad = stat["bad"].sum()

        ratio_good = pd.Series(list(map(lambda x: float(sum(x["good"])) / total_good, lst_df)))
        ratio_bad = pd.Series(list(map(lambda x: float(sum(x["bad"])) / total_bad, lst_df)))

        # monotonous property
        lst_woe = list(np.log(ratio_good / ratio_bad))
        if sorted(lst_woe) != lst_woe and sorted(lst_woe, reverse=True) != lst_woe:
            return None

        lst_iv = (ratio_good - ratio_bad) * np.log(ratio_good / ratio_bad)
        if np.inf in list(lst_iv) or -np.inf in list(lst_iv):
            return None
        else:
            return sum(lst_iv)

    def combine_helper(self, stat, nbins, knots):
        lst_knots = list(combinations(knots, nbins - 1))

        knots_list = list(map(lambda x: sorted(x + (0, len(stat) - 1)), lst_knots))
        lst_iv = list(map(lambda x: self.eval_iv_mono(stat, x), knots_list))

        lst_iv_filted = list(filter(lambda x: x is not None, lst_iv))
        if len(lst_iv_filted) == 0:
            return None
        else:
            if len(self.__find_index(lst_iv, max(lst_iv_filted))) > 0:
                target_index = self.__find_index(lst_iv, max(lst_iv_filted))[0]
                return knots_list[target_index]
            else:
                return None

    def combine_bins(self, stat, max_nbins, knots):
        max_nbins = min(max_nbins, len(knots) + 1)
        if max_nbins == 1:
            return [0, len(stat) - 1]
        for cur_nbins in sorted(range(2, max_nbins + 1), reverse=True):
            new_knots = self.combine_helper(stat, cur_nbins, knots)

            if new_knots is not None:
                return new_knots
        print("no available bins with mono contrain")

        return [0, len(stat) - 1]

    def __refit_num(self, df, c, ws):
        df_missing, df_data = self.split_data_by_missing(df, c)

        stat_missing = self.stat_group(df_missing, c, "label")
        df_data.loc[:, c] = df_data.loc[:, c].apply(lambda x: round(x, 4))

        df_data["bin"] = pd.cut(df_data[c], ws, right=True)

        stat_data = (
            df_data.groupby("bin")
            .agg(
                good=("label", lambda x: x.count() - x.sum()),
                bad=("label", lambda x: x.sum()),
            )
            .reset_index()
        )
        stat_data["var"] = c

        print(stat_missing)
        print(stat_data)

        df_bin = self.stat_bin_info_update_bin(stat_missing, stat_data)
        return df_bin

    def update(self, df_xtrain, df_ytrain, ws, **kwargs):
        origin_feature_missing_logic = self.missing_logic[df_xtrain.name]
        feature_missing_logic = kwargs.get("missing_logic", origin_feature_missing_logic)
        if feature_missing_logic is not None:
            self.missing_logic[df_xtrain.name] = feature_missing_logic

        df_train = pd.DataFrame({df_xtrain.name: df_xtrain, "label": df_ytrain})

        df_missing, df_data = self.split_data_by_missing(df_train, df_xtrain.name)
        stat_missing = self.stat_group(df_missing, df_xtrain.name, "label")

        bin_info = self.bin_info
        bin_rest = bin_info[bin_info["var"] != df_xtrain.name]
        bin_update = self.__refit_num(df_train, df_xtrain.name, ws)

        if len(stat_missing) > 0:
            bin_total = bin_update
            woe_max = bin_total["woe"].max()
            woe_min = bin_total["woe"].min()
        else:
            bin_missing = pd.DataFrame(
                [
                    {
                        "var": df_xtrain.name,
                        "type": "num_missing",
                        "bin": -990000,
                        "min": -990000,
                        "max": -990000,
                        "woe": -np.inf,
                        "iv": 0,
                        "total": 0,
                        "ratio": 0,
                        "bad": 0,
                        "bad_rate": 0,
                    }
                ]
            )
            bin_total = pd.concat([bin_update, bin_missing], axis=0).reset_index(drop=True)
            woe_max = bin_update["woe"].max()
            woe_min = bin_update["woe"].min()

        bin_total["bin"] = bin_total["bin"].apply(lambda x: str(x))
        for idx, row in bin_total.iterrows():
            if row["type"] == "num_normal":
                v = f"{idx:02}.{row['bin']}"
            else:
                v = f"{idx:02}.{{{row['bin']}}}"

            bin_total.at[idx, "bin"] = v

        bin_total["woe_max"] = woe_max
        bin_total["woe_min"] = woe_min

        self.bin_info = pd.concat([bin_rest, bin_total])
        self.bin_info = self.__adjust_woe(self.bin_info)

        return self

    def __adjust_woe(self, bin_info):
        if bin_info is None or len(bin_info) <= 0:
            return bin_info

        mask = (
            ((bin_info["ratio"] < self.min_bin_rate) | (bin_info["total"] < self.min_bin_size))
            & (bin_info["woe"] != -np.inf)
            & (bin_info["type"].isin(["num_normal", "cat_normal"]))
        )
        bin_info.loc[mask, "woe"] = 0

        mask = (
            (bin_info["bad"] < self.min_missing_bad_cnt)
            & (bin_info["woe"] != -np.inf)
            & (bin_info["type"].isin(["num_missing", "cat_missing"]))
        )
        bin_info.loc[mask, "woe"] = 0

        for c in self.feature_list:
            if c in self.categorical_features:
                mask = (bin_info["var"] == c) & (bin_info["type"] == "cat_missing") & (bin_info["woe"] == -np.inf)
                if self.missing_logic[c] == "high_risk":
                    val = bin_info.loc[
                        (bin_info["var"] == c) & (bin_info["type"] == "cat_normal"),
                        "woe",
                    ].min()
                elif self.missing_logic[c] == "low_risk":
                    val = bin_info.loc[
                        (bin_info["var"] == c) & (bin_info["type"] == "cat_normal"),
                        "woe",
                    ].max()
                else:
                    val = 0
                bin_info.loc[mask, "woe"] = val
            else:
                mask = (bin_info["var"] == c) & (bin_info["type"] == "num_missing") & (bin_info["woe"] == -np.inf)
                if self.missing_logic[c] == "high_risk":
                    val = bin_info.loc[
                        (bin_info["var"] == c) & (bin_info["type"] == "num_normal"),
                        "woe",
                    ].min()
                elif self.missing_logic[c] == "low_risk":
                    val = bin_info.loc[
                        (bin_info["var"] == c) & (bin_info["type"] == "num_normal"),
                        "woe",
                    ].max()
                else:
                    val = 0
                bin_info.loc[mask, "woe"] = val

        bin_info["woe_raw"] = bin_info["woe"]

        return bin_info

    def update_by_adding_cutoff(self, df_xtrain, df_ytrain, cutoff, **kwargs):
        bin_list = self.export()[df_xtrain.name]["data"]
        bin_list.append(cutoff)
        bin_list = sorted(set(bin_list))
        self.update(df_xtrain, df_ytrain, bin_list, **kwargs)
        return self

    def export(self):
        dict_ws = dict()
        for c in self.feature_list:
            dict_ws[c] = dict()

            if c in self.categorical_features:
                bin_missing = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "cat_missing")]
                bin_data = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "cat_normal")]

                dict_ws[c]["missing"] = sorted(bin_missing["min"].values)
                dict_ws[c]["data"] = sorted(bin_data["min"].values)

            else:
                bin_missing = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "num_missing")]
                bin_data = self.bin_info[(self.bin_info["var"] == c) & (self.bin_info["type"] == "num_normal")]

                dict_ws[c]["missing"] = sorted([float(v) for v in bin_missing["min"].values])
                dict_ws[c]["data"] = sorted(bin_data["min"].values)
                dict_ws[c]["data"].append(np.inf)
        return dict_ws

    def set_woe(self, var, bin_name, woe):
        self.bin_info.loc[(self.bin_info["var"] == var) & (self.bin_info["bin"] == bin_name), "woe"] = woe


def main():
    pass


if __name__ == "__main__":
    main()
