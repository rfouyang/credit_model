from pathlib import Path

import config
from util import data_helper, woe_helper, report_helper
from projects.ccard import profile


def train_woe_encoding(df_feature):
    features = sorted(set(df_feature.columns) - {profile.pk, profile.label, profile.sample_type})
    df_train = df_feature[df_feature['sample_type'] == 'train']

    woe_dt = woe_helper.WOE()
    woe_dt.fit(df_train[features + [profile.label]], profile.label)
    data_helper.Data.dump('woe_dt_model', woe_dt, prefix=profile.prefix)

    df_bin = woe_dt.transform(df_feature, bin_only=True)
    df_woe = woe_dt.transform(df_feature)

    df_report = report_helper.FTReport.get_report(df_bin, features, profile.label)
    df_eval = report_helper.FTReport.eval_metrics(df_woe, features, profile.label)
    df_report = df_report.merge(df_eval, on='feature', how='left')
    df_report.to_csv(Path(config.DATA_DIR, profile.prefix, 'report.csv'), index=None)


def encoding_using_woe(df_feature):
    woe_dt = data_helper.Data.load('woe_dt_model', prefix=profile.prefix)
    df_woe = woe_dt.transform(df_feature)
    data_helper.Data.dump('df_woe', df_woe, prefix=profile.prefix)


def main():
    df_feature = data_helper.Data.load('df_feature', prefix=profile.prefix)
    train_woe_encoding(df_feature)
    encoding_using_woe(df_feature)


if __name__ == '__main__':
    main()
