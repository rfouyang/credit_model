from util import data_helper, lgbm_helper, report_helper
from projects.acard import profile


def train_gbdt(df_woe):
    features = sorted(set(df_woe.columns) - {profile.pk, profile.label, profile.sample_type})
    df_train = df_woe[df_woe['sample_type'] == 'train']
    df_test = df_woe[df_woe['sample_type'] == 'test']

    gbdt = lgbm_helper.LightGBM()
    n_estimators = 500
    params = {
        "learning_rate": 0.1,
        "max_depth": 2,
        "num_leaves": 4,
        "min_child_samples": 20,
        "subsample": 1,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 10,
        "reg_lambda": 10,
    }

    gbdt.select_and_fit(
        df_train[features], df_train[profile.label],
        df_xvalid=df_test[features], df_yvalid=df_test[profile.label],
        n_estimators=n_estimators, params=params
    )

    df_woe['gbdt_prob'] = gbdt.predict(df_woe)
    df_report = report_helper.ModelReport.get_report(df_woe, profile.sample_type, 'gbdt_prob', profile.label)
    print(df_report)
    print(gbdt.get_summary())
    print(gbdt.selected_features)

    data_helper.Data.dump('gbdt_model', gbdt, prefix=profile.prefix)


def main():
    df_woe = data_helper.Data.load('df_woe', prefix=profile.prefix)
    train_gbdt(df_woe)


if __name__ == '__main__':
    main()
