from util import data_helper, bm_helper, report_helper
from projects.acard import profile


def train_benchmark(df_woe):
    features = sorted(set(df_woe.columns) - {profile.pk, profile.label, profile.sample_type})
    df_train = df_woe[df_woe['sample_type'] == 'train']
    logit = bm_helper.Logit()
    logit.select_and_fit(df_train[features], df_train[profile.label])
    df_woe['logit_prob'] = logit.predict(df_woe)
    df_report = report_helper.ModelReport.get_report(df_woe, profile.sample_type, 'logit_prob', profile.label)
    print(df_report)
    print(logit.get_summary())
    print(logit.selected_features)

    data_helper.Data.dump('logit_model', logit, prefix=profile.prefix)


def main():
    df_woe = data_helper.Data.load('df_woe', prefix=profile.prefix)
    train_benchmark(df_woe)


if __name__ == '__main__':
    main()
