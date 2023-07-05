import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toad
from sklearn.base import TransformerMixin
from tqdm import tqdm

warnings.simplefilter(action="ignore")


class WOE(TransformerMixin):
    def __init__(self):
        self.combiner = toad.transform.Combiner()
        self.woe_encoder = toad.transform.WOETransformer()
        self.selected_features = list()

    def fit(self, df_feature, label, **kwargs):
        exclude = kwargs.get("exclude", list())

        df_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
        selected_features = sorted(set(df_feature.columns) - set(label) - set(exclude))

        method = kwargs.get("method", "dt")
        if method in ["quantile", "step", "kmeans"]:
            dict_bin = dict()
            for c in tqdm(selected_features):
                try:
                    self.combiner.fit(
                        df_feature[[c, label]],
                        y=label,
                        method=method,
                        n_bins=kwargs.get("n_bins", 5),
                        empty_separate=True,
                    )
                    rule = self.combiner.export()
                    dict_bin.update(rule)
                except Exception as e:
                    print(f"{e}")
                    print(f"feature {c} is failed")
                    traceback.print_exc()
            self.combiner.load(dict_bin)

        elif method in ["chi", "dt"]:
            dict_bin = dict()
            for c in tqdm(selected_features):
                try:
                    self.combiner.fit(
                        df_feature[[c, label]],
                        y=label,
                        method=method,
                        min_samples=kwargs.get("min_samples", 0.1),
                        n_bins=kwargs.get("n_bins", 5),
                        empty_separate=True,
                    )
                    rule = self.combiner.export()
                    dict_bin.update(rule)
                except Exception as e:
                    print(f"{e}")
                    print(f"feature {c} is failed")
                    traceback.print_exc()
            self.combiner.load(dict_bin)
        else:
            raise ValueError("bin method should be quantile, step, kmeans, chi, dt")

        print("finish combiner fit")
        df_bin = self.combiner.transform(df_feature, labels=True)
        print("finish combiner transform")

        dict_woe = dict()
        for c in tqdm(list(dict_bin.keys())):
            self.woe_encoder.fit(df_bin[[c, label]], df_bin[label], exclude=[label])
            rule = self.woe_encoder.export()
            dict_woe.update(rule)
        self.woe_encoder.load(dict_woe)
        print("finish WOE fit")

        self.selected_features = list(self.combiner.export().keys())

    def transform(self, df_feature, **kwargs):
        bin_only = kwargs.get("bin_only", False)
        cols = self.selected_features

        df_bin = df_feature.copy()
        if bin_only is True:
            for c in tqdm(cols):
                df_bin[c] = self.combiner.transform(df_bin[c], labels=True)
        else:
            for c in tqdm(cols):
                df_bin[c] = self.combiner.transform(df_bin[c], labels=True)
            for c in tqdm(cols):
                df_bin[c] = self.woe_encoder.transform(df_bin[c])
        return df_bin

    def update_rule(self, rule, df_feature, label, **kwargs):
        self.combiner.update(rule)
        print("finish update rule")

        df_bin = self.combiner.transform(df_feature, labels=True)
        print("finish combiner transform")

        dict_woe = self.woe_encoder.export()
        for c in tqdm(list(rule.keys())):
            self.woe_encoder.fit(df_bin[[c, label]], df_bin[label], exclude=[label])
            rule = self.woe_encoder.export()
            dict_woe.update(rule)
        self.woe_encoder.load(dict_woe)
        print("finish WOE refit")

        self.selected_features = list(self.combiner.export().keys())