import numpy as np
import pandas as pd
import statsmodels.api as sm
import toad
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from termcolor import cprint
from tqdm import tqdm

from util.metric_helper import Metric


class Logit(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
        self.summary = None
        self.selected_features = None

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        maxiter = kwargs.get("maxiter", 100)
        self.selected_features = list(df_xtrain.columns)
        df_xtrain_const = sm.add_constant(df_xtrain[self.selected_features])

        try:
            self.model = sm.Logit(df_ytrain, df_xtrain_const).fit(method="newton", maxiter=maxiter)
        except Exception as e:
            cprint(
                "warning:  exist strong correlated features, "
                "got singular matrix in linear model, retry bfgs method instead.",
                "red",
            )
            self.model = sm.Logit(df_ytrain, df_xtrain_const).fit(method="bfgs", maxiter=maxiter)

        # prepare model result
        self.summary = pd.DataFrame(
            {
                "var": df_xtrain_const.columns.tolist(),
                "coef": self.model.params,
                "std_err": [round(v, 2) for v in self.model.bse],
                "z": [round(v, 2) for v in self.model.tvalues],
                "pvalue": [round(v, 2) for v in self.model.pvalues],
            }
        )

        self.summary["std_var"] = df_xtrain.std()
        self.summary["std_var"] = self.summary["std_var"].apply(lambda x: round(x, 2))

        self.summary["feature_importance"] = abs(self.summary["coef"]) * self.summary["std_var"]
        self.summary["feature_importance"] /= self.summary["feature_importance"].sum()
        self.summary["feature_importance"] = self.summary["feature_importance"].apply(lambda x: round(x, 2))

        return self

    def select_and_fit(self, df_xtrain, df_ytrain, **kwargs):
        maxiter = kwargs.get("maxiter", 100)

        positive_coef = kwargs.get("positive_coef", True)
        n_jobs = kwargs.get("n_jobs", -1)

        self.selected_features = list(df_xtrain.columns)
        label = df_ytrain.name

        # remove variable with toad
        df_train, removed = toad.selection.select(
            pd.concat([df_xtrain, df_ytrain], axis=1),
            target=label,
            empty=0.9,
            iv=0.02,
            corr=0.7,
            return_drop=True,
        )
        self.selected_features = sorted(set(df_train.columns) - {label})
        print(f"removed {len(removed['empty']) + len(removed['iv']) + len(removed['corr'])} features")
        print(removed)

        df_xtrain, df_ytrain = df_train[self.selected_features], df_train[label]

        # remove variable with inconsistent trend between woe and coefficient
        coef_selector = CoefSelector()
        coef_selector.fit(
            df_xtrain[self.selected_features],
            df_ytrain,
            positive_coef=positive_coef,
            remove_method="iv",
            df_iv=None,
            n_jobs=n_jobs,
        )
        self.selected_features = coef_selector.selected_features
        print(self.selected_features)

        # remove variable with insignificant p value
        pvalue_selector = PValueSelector()
        pvalue_selector.fit(df_xtrain[self.selected_features], df_ytrain, pvalue_threshold=0.05)
        self.selected_features = pvalue_selector.selected_features
        self.pvalue_selector = pvalue_selector
        print(self.selected_features)

        self.fit(df_xtrain, df_ytrain, **kwargs)

        return self

    def predict(self, df_xtest, **kwargs):
        df_xtest = sm.add_constant(df_xtest[self.selected_features])
        df_xtest["const"] = 1
        return self.model.predict(df_xtest[["const"] + self.selected_features])

    def predict_proba(self, df_xtest, **kwargs):
        df_xtest = sm.add_constant(df_xtest[self.selected_features])
        df_xtest["const"] = 1
        yprob = self.model.predict(df_xtest[["const"] + self.selected_features])
        res = np.zeros((len(df_xtest), 2))
        res[:, 1] = yprob
        res[:, 0] = 1 - yprob
        return res

    def get_summary(self):
        df_summary = self.summary
        return df_summary

    def get_importance(self):
        df_importance = self.summary.drop("const", axis=0)
        return df_importance.sort_values(by="feature_importance", ascending=False)


class CoefSelector(TransformerMixin):
    def __init__(self):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        positive_coef = kwargs.get("positive_coef", False)
        remove_method = kwargs.get("remove_method", "iv")

        feature_list = sorted(kwargs.get("feature_list", df_xtrain.columns.tolist()))

        self.selected_features = feature_list
        self.detail = list()

        lst_iv = list()
        for c in tqdm(feature_list, position=0, leave=True):
            iv = Metric.get_iv(df_xtrain[c], df_ytrain)
            lst_iv.append(iv)
        df_iv = pd.DataFrame({"var": feature_list, "iv": lst_iv})
        df_iv = df_iv[["var", "iv"]]

        while True:
            print(f"removed {len(self.removed_features)} features")
            print(self.removed_features)
            model = Logit()
            model.fit(df_xtrain[self.selected_features], df_ytrain)

            if remove_method == "feature_importance":
                df_res = model.get_importance()[["var", "coef", "pvalue", "feature_importance"]]
                df_res = df_res.reset_index(drop=True)
                self.detail.append(df_res)
            else:
                df_res = model.get_importance()[["var", "coef", "pvalue"]]
                df_res = df_res.reset_index(drop=True)
                df_res = df_res.merge(df_iv, on=["var"], how="left")
                self.detail.append(df_res)

            if df_res["pvalue"].isnull().sum() != 0:
                df_remove = df_res.loc[(df_res["pvalue"].isnull()), :]
                df_remove = df_remove.sort_values(by=f"{remove_method}", ascending=True)
                df_remove = df_remove.reset_index(drop=True)
                remove_var = df_remove.loc[0, "var"]
                self.selected_features.remove(remove_var)
                self.removed_features.append(remove_var)
            else:
                if positive_coef is True:
                    df_res["coef"] = -df_res["coef"]

                df_remove = df_res.loc[(df_res["coef"] >= 0), :]
                if len(df_remove) != 0:
                    df_remove = df_remove.sort_values(by=f"{remove_method}", ascending=True)
                    df_remove = df_remove.reset_index(drop=True)
                    remove_var = df_remove.loc[0, "var"]
                    self.selected_features.remove(remove_var)
                    self.removed_features.append(remove_var)
                else:
                    break

            if len(self.selected_features) == 0:
                break

        return self

    def transform(self, df_xtest, **kwargs):
        feature_list = kwargs.get("feature_list", df_xtest.columns.tolist())
        feature_list = sorted(set(feature_list) & set(self.selected_features))
        return df_xtest[feature_list]

    def summary(self):
        print("\nselected features:")
        print(self.selected_features)
        print("\nremoved features:")
        print(self.removed_features)
        print("\nsummary")
        for idx, df in enumerate(self.detail):
            print("iter:", idx)
            print(self.detail[idx])


class PValueSelector(TransformerMixin):
    def __init__(self):
        self.detail = None
        self.selected_features = list()
        self.removed_features = list()

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        pvalue_threshold = kwargs.get("pvalue_threshold", 0.05)
        feature_list = sorted(kwargs.get("feature_list", df_xtrain.columns.tolist()))

        self.selected_features = feature_list
        self.detail = list()

        while True:
            print(f"removed {len(self.removed_features)} features")
            print(self.removed_features)
            model = Logit()
            model.fit(df_xtrain[self.selected_features], df_ytrain)

            df_res = model.get_importance()[["var", "coef", "pvalue"]]
            df_res = df_res.reset_index(drop=True)
            self.detail.append(df_res)

            df_remove = df_res.loc[(df_res["pvalue"] > pvalue_threshold), :]
            if len(df_remove) != 0:
                df_remove = df_remove.sort_values(by="pvalue", ascending=False)
                df_remove = df_remove.reset_index()
                remove_var = df_remove.loc[0, "var"]
                self.selected_features.remove(remove_var)
                self.removed_features.append(remove_var)
            else:
                break

            if len(self.selected_features) == 0:
                break
        return self

    def transform(self, df_xtest, **kwargs):
        feature_list = kwargs.get("feature_list", df_xtest.columns.tolist())
        feature_list = sorted(set(feature_list) & set(self.selected_features))
        return df_xtest[feature_list]

    def summary(self):
        print("\nselected features:")
        print(self.selected_features)
        print("\nremoved features:")
        print(self.removed_features)
        print("\nsummary")
        for idx, df in enumerate(self.detail):
            print("iter:", idx)
            print(self.detail[idx])
