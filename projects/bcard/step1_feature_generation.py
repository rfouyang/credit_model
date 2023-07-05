from pathlib import Path
import pandas as pd

import config
from util import data_helper
from projects.bcard import profile


def get_payment_ratio(r):
    if r['pre_outstanding'] <= 0:
        return 1
    elif r['pre_outstanding'] <= 10:
        if r['payment'] == 0:
            return 0
        else:
            return 1
    else:
        return r['payment'] / r['pre_outstanding']


def get_max_dpd(r):
    if r['is_m3'].sum() > 0:
        return 3
    elif r['is_m2'].sum() > 0:
        return 2
    elif r['is_m1'].sum() > 0:
        return 1
    else:
        return 0


def load_data():
    fp_user = Path(config.DATA_DIR, 'bcard_data', 'user.csv')
    fp_repayment = Path(config.DATA_DIR, 'bcard_data', 'repayment.csv')

    df_user = pd.read_csv(fp_user, index_col=None)
    df_repayment = pd.read_csv(fp_repayment, index_col=None)

    df_data = df_user.merge(df_repayment, on=profile.pk, how='left')
    return df_data


def preprocess(df_data):
    # utility feature reprocessing
    df_data['ur'] = df_data['spend'] / df_data['limit_amount']
    df_data['pre_ur'] = df_data.sort_values("month_gap").groupby(profile.pk)["ur"].shift(1)
    df_data['ur_chg'] = df_data['ur'] / df_data['pre_ur']

    # payment feature reprocessing
    df_data['payment_ratio'] = df_data.apply(lambda r: get_payment_ratio(r), axis=1)

    return df_data


def get_overdue_features(df_data, offset):
    df_sub = df_data.loc[df_data['month_gap'] <= offset]
    df_ft = df_sub.groupby(profile.pk).agg(
        ft_overdue_m1_cnt=('is_m1', 'sum'),
        ft_overdue_m2_cnt=('is_m2', 'sum'),
        ft_overdue_m3_cnt=('is_m3', 'sum')
    )

    df_ft['ft_overdue_max_dpd_in_month'] = df_sub.groupby(profile.pk).apply(lambda r: get_max_dpd(r))

    dict_name = dict()
    for c in df_ft.columns:
        if 'ft_' in c:
            dict_name[c] = f'{c}_{offset}m'
    df_ft = df_ft.rename(columns=dict_name)
    return df_ft


def get_utilization_features(df_data, offset):
    df_sub = df_data.loc[df_data['month_gap'] <= offset]
    df_ft = df_sub.groupby(profile.pk).agg(
        ft_ur_mean=('ur', 'mean'),
        ft_ur_max=('ur', 'max'),
        ft_ur_increase_month_cnt=('ur_chg', lambda x: len([v for v in x if v > 1]))
    )

    dict_name = dict()
    for c in df_ft.columns:
        if 'ft_' in c:
            dict_name[c] = f'{c}_{offset}m'
    df_ft = df_ft.rename(columns=dict_name)
    return df_ft


def get_payment_features(df_data, offset):
    df_sub = df_data.loc[df_data['month_gap'] <= offset]
    df_cum = df_sub.groupby(profile.pk).agg(
        payment=('payment', 'sum'),
        pre_outstanding=('pre_outstanding', 'sum')
    )

    df_ft = df_sub.groupby(profile.pk).agg(
        ft_payment_ratio_mean=('payment_ratio', 'mean'),
        ft_payment_ratio_min=('payment_ratio', 'min'),
        ft_payment_ratio_max=('payment_ratio', 'max')
    )

    df_ft['ft_payment_cum_ratio'] = df_cum.apply(lambda r: get_payment_ratio(r), axis=1)

    dict_name = dict()
    for c in df_ft.columns:
        if 'ft_' in c:
            dict_name[c] = f'{c}_{offset}m'
    df_ft = df_ft.rename(columns=dict_name)
    return df_ft


def get_features(df_data):
    df_feature = df_data[[profile.pk, profile.label]].copy(deep=True)
    for offset in [3, 6, 12]:
        print(f"generating overdue features for past {offset} months")
        df_ft = get_overdue_features(df_data, offset)
        df_feature = df_feature.merge(df_ft, on=profile.pk, how='left')

    for offset in [3, 6, 12]:
        print(f"generating utilization features for past {offset} months")
        df_ft = get_utilization_features(df_data, offset)
        df_feature = df_feature.merge(df_ft, on=profile.pk, how='left')

    for offset in [3, 6, 12]:
        print(f"generating payment features for past {offset} months")
        df_ft = get_payment_features(df_data, offset)
        df_feature = df_feature.merge(df_ft, on=profile.pk, how='left')

    return df_feature


def main():
    df_data = load_data()
    df_data = preprocess(df_data)
    df_feature = get_features(df_data)

    df_label = data_helper.Data.train_test_split(df_feature, profile.pk, profile.label)

    df_feature = df_label[[profile.pk, profile.sample_type]].merge(df_feature, on=profile.pk, how='left')
    data_helper.Data.dump('df_feature', df_feature, prefix=profile.prefix)


if __name__ == '__main__':
    main()
