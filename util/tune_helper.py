from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import StratifiedKFold

from util import metric_helper


class GridSearch:
    def __init__(self):
        self.best_params = None
        self.best_estimator = None
        self.results = list()

    def fit(self, df_xtrain, df_ytrain, df_xtest, df_ytest, param_grid, **kwargs):

        lst_params = self.generate_param_grid(param_grid)

        for param_grid in lst_params:
            model = self.init_model(param_grid)
            cv_res = self.get_cv_results(model, df_xtrain, df_ytrain)
            model, train_res = self.get_results(model, df_xtrain, df_ytrain, df_xtest, df_ytest)

            param_grid["best_iteration"] = model.best_iteration_
            res = param_grid
            res.update(cv_res)
            res.update(train_res)
            self.results.append(res)

            print("")
            print(param_grid)
            print(cv_res)
            print(train_res)
            print("")

            if self.best_params is None:
                self.best_params = param_grid
                self.best_estimator = model
            elif res["cv_valid_avg_auc"] == max([res["cv_valid_avg_auc"] for res in self.results]):
                self.best_params = param_grid
                self.best_estimator = model
            else:
                continue

        self.results = pd.concat([pd.DataFrame(res) for res in self.results], axis=0).reset_index(drop=True)
        return self

    @classmethod
    def generate_param_grid(cls, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]

        lst_params = list()
        for p in param_grid:
            items = sorted(p.items())
            keys, values = zip(*items)

            for v in product(*values):
                params = dict(zip(keys, v))
                lst_params.append(params)
        return lst_params

    @classmethod
    def init_model(cls, param_grid):
        model = lgb.LGBMClassifier(
            max_depth=param_grid["max_depth"],
            num_leaves=param_grid["num_leaves"],
            reg_alpha=param_grid["reg_alpha"],
            reg_lambda=param_grid["reg_lambda"],
            learning_rate=param_grid["learning_rate"],
            n_estimators=param_grid["n_estimators"],
            min_child_samples=param_grid["min_child_samples"],
            class_weight="balanced",
            importance_type="gain",
            n_jobs=16,
            random_state=1024,
        )
        return model

    @classmethod
    def get_cv_results(cls, model, df_xtrain, df_ytrain):
        cv_res = dict()

        lst_train_auc, lst_train_ks, lst_valid_auc, lst_valid_ks = (
            list(),
            list(),
            list(),
            list(),
        )
        kf = StratifiedKFold(n_splits=5, random_state=1024, shuffle=True)
        for idx_train, idx_valid in kf.split(df_xtrain, df_ytrain):
            df_xtrain_train = df_xtrain.loc[idx_train, :]
            df_ytrain_train = df_ytrain.loc[idx_train, :]
            df_xtrain_valid = df_xtrain.loc[idx_valid, :]
            df_ytrain_valid = df_ytrain.loc[idx_valid, :]

            model.fit(
                df_xtrain_train,
                df_ytrain_train,
                eval_metric="auc",
                eval_set=[
                    (df_xtrain_train, df_ytrain_train),
                    (df_xtrain_valid, df_ytrain_valid),
                ],
                early_stopping_rounds=20,
                verbose=500,
            )

            df_ypred_train = model.predict_proba(df_xtrain_train)[:, 1]
            df_ypred_valid = model.predict_proba(df_xtrain_valid)[:, 1]

            lst_train_auc.append(metric_helper.Metric.get_auc(df_ytrain_train, df_ypred_train))
            lst_train_ks.append(metric_helper.Metric.get_ks(df_ytrain_train, df_ypred_train))
            lst_valid_auc.append(metric_helper.Metric.get_auc(df_ytrain_valid, df_ypred_valid))
            lst_valid_ks.append(metric_helper.Metric.get_ks(df_ytrain_valid, df_ypred_valid))

        cv_res["cv_train_auc"] = [[round(x, 4) for x in lst_train_auc]]
        cv_res["cv_valid_auc"] = [[round(x, 4) for x in lst_valid_auc]]
        cv_res["cv_train_ks"] = [[round(x, 4) for x in lst_train_ks]]
        cv_res["cv_valid_ks"] = [[round(x, 4) for x in lst_valid_ks]]

        cv_res["cv_train_avg_auc"] = [np.mean(lst_train_auc)]
        cv_res["cv_valid_avg_auc"] = [np.mean(lst_valid_auc)]
        cv_res["cv_train_avg_ks"] = [np.mean(lst_train_ks)]
        cv_res["cv_valid_avg_ks"] = [np.mean(lst_valid_ks)]
        cv_res["cv_ks_gap"] = [abs(np.mean(lst_valid_ks) - np.mean(lst_train_ks))]

        return cv_res

    @classmethod
    def get_results(cls, model, df_xtrain, df_ytrain, df_xtest, df_ytest, **kwargs):
        res = {
            "train_auc": list(),
            "test_auc": list(),
            "train_ks": list(),
            "test_ks": list(),
            "ks_gap": list(),
        }

        model.fit(
            df_xtrain,
            df_ytrain,
            eval_metric="auc",
            eval_set=[(df_xtrain, df_ytrain), (df_xtest, df_ytest)],
            early_stopping_rounds=20,
            verbose=500,
        )

        df_ypred_train = model.predict_proba(df_xtrain)[:, 1]
        df_ypred_test = model.predict_proba(df_xtest)[:, 1]

        res["train_auc"].append(metric_helper.Metric.get_auc(df_ytrain, df_ypred_train))
        res["test_auc"].append(metric_helper.Metric.get_auc(df_ytest, df_ypred_test))
        res["train_ks"].append(metric_helper.Metric.get_ks(df_ytrain, df_ypred_train))
        res["test_ks"].append(metric_helper.Metric.get_ks(df_ytest, df_ypred_test))
        res["ks_gap"].append(
            abs(
                metric_helper.Metric.get_ks(df_ytest, df_ypred_test)
                - metric_helper.Metric.get_ks(df_ytrain, df_ypred_train)
            )
        )

        return model, res
