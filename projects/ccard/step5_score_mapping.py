from util import data_helper, score_helper, scorebin_helper, report_helper
from projects.ccard import profile


def score_mapping(df_woe):
    sb = score_helper.Score()
    sb.fit(df_woe['prob'], df_woe[profile.label])
    df_woe['score'] = sb.transform(df_woe['prob'])
    df_report = report_helper.ModelReport.get_report(df_woe, profile.sample_type, 'score', profile.label)
    print(df_report)

    data_helper.Data.dump('df_score', df_woe, prefix=profile.prefix)


def stat_score_info(df_score):
    df_score = scorebin_helper.ScoreBin.bin_megascore(df_score, 'score', bin_name='bin')
    df_stat = df_score.groupby('bin').agg(
        total=(profile.label, 'count'),
        bad=(profile.label, 'sum'),
        bad_rate=(profile.label, 'mean')
    )
    print(df_stat)


def main():
    df_woe = data_helper.Data.load('df_woe', prefix=profile.prefix)
    model = data_helper.Data.load('gbdt_model', prefix=profile.prefix)
    df_woe['prob'] = model.predict(df_woe[model.selected_features])
    score_mapping(df_woe)

    df_score = data_helper.Data.load('df_score', prefix=profile.prefix)
    stat_score_info(df_score)


if __name__ == '__main__':
    main()
