import lightgbm as lgb
import pandas as pd
import toad
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class LightGBM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
        self.summary = None
        self.selected_features = None

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        n_estimators = kwargs.get("n_estimators", 200)
        default_param = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 800,
            "subsample": 1,
            "subsample_freq": 0,
            "colsample_bytree": 1,
            "reg_alpha": 157,
            "reg_lambda": 500,
        }

        self.selected_features = list(df_xtrain.columns)
        params = kwargs.get("params", default_param)

        self.model = lgb.LGBMClassifier(
            **params,
            n_estimators=n_estimators,
            objective="cross_entropy",
            class_weight="balanced",
            importance_type="gain",
            boosting_type="gbdt",
            silent=True,
            n_jobs=8,
            random_state=417,
        )

        eval_set = [(df_xtrain[self.selected_features], df_ytrain)]
        if "df_xvalid" in kwargs:
            df_xvalid, df_yvalid = kwargs.get("df_xvalid"), kwargs.get("df_yvalid")
            eval_set.append((df_xvalid[self.selected_features], df_yvalid))

        self.model.fit(
            df_xtrain[self.selected_features],
            df_ytrain,
            eval_set=eval_set,
            verbose=50,
            early_stopping_rounds=8000,
        )

        return self

    def select_and_fit(self, df_xtrain, df_ytrain, **kwargs):
        label = df_ytrain.name
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

        self.fit(df_xtrain, df_ytrain, **kwargs)

        return self

    def predict(self, df_xtest, **kwargs):
        yprob = self.model.predict_proba(df_xtest[self.selected_features])[:, 1]
        return yprob

    def predict_proba(self, df_xtest, **kwargs):
        yprob = self.model.predict_proba(df_xtest[self.selected_features])
        return yprob

    def get_summary(self):
        lgb.plot_metric(self.model)

        # lgb.create_tree_digraph(model.model.booster_, tree_index=0,
        # show_info=['leaf_count', 'leaf_weight', 'data_percentage', 'split_gain', 'internal_value'])
        df_tree = self.model.booster_.trees_to_dataframe()
        return df_tree

    def get_importance(self):
        df_importance = pd.DataFrame(
            {
                "feature": self.model.feature_name_,
                "split": self.model.booster_.feature_importance(importance_type='split'),
                "gain": self.model.booster_.feature_importance(importance_type='gain')
            }
        )
        return df_importance.sort_values(by="split", ascending=False)


class LGBR(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None
        self.summary = None
        self.selected_features = None

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        n_estimators = kwargs.get("n_estimators", 800)
        default_param = {
            "learning_rate": 0.01,
            "max_depth": 4,
            "num_leaves": 16,
            "min_child_samples": 10,
            "subsample": 1,
            "subsample_freq": 0,
            "colsample_bytree": 1,
            "reg_alpha": 500,
            "reg_lambda": 100,
        }

        self.selected_features = list(df_xtrain.columns)
        params = kwargs.get("params", default_param)

        self.model = lgb.LGBMRegressor(
            **params,
            n_estimators=n_estimators,
            objective="regression",
            class_weight="balanced",
            importance_type="gain",
            boosting_type="gbdt",
            silent=True,
            n_jobs=8,
            random_state=417,
        )

        eval_set = [(df_xtrain[self.selected_features], df_ytrain)]
        if "df_xvalid" in kwargs:
            df_xvalid, df_yvalid = kwargs.get("df_xvalid"), kwargs.get("df_yvalid")
            eval_set.append((df_xvalid[self.selected_features], df_yvalid))

        self.model.fit(
            df_xtrain[self.selected_features],
            df_ytrain,
            eval_set=eval_set,
            verbose=50,
            early_stopping_rounds=8000,
        )

        return self

    def predict(self, df_xtest, **kwargs):
        yprob = self.model.predict(df_xtest[self.selected_features])
        return yprob

    def predict_proba(self, df_xtest, **kwargs):
        yprob = self.model.predict_proba(df_xtest[self.selected_features])
        return yprob

    def get_summary(self):
        lgb.plot_metric(self.model)

        # lgb.create_tree_digraph(model.model.booster_, tree_index=0,
        # show_info=['leaf_count', 'leaf_weight', 'data_percentage', 'split_gain', 'internal_value'])
        df_tree = self.model.booster_.trees_to_dataframe()
        return df_tree

    def get_importance(self):
        df_importance = pd.DataFrame(
            {
                "feature": self.model.feature_name_,
                "importance": self.model.feature_importances_,
            }
        )
        return df_importance.sort_values(by="importance", ascending=False)
