from pathlib import Path
import pandas as pd

import config
from util import data_helper
from projects.acard import profile



def load_data():
    fp_data = Path(config.DATA_DIR, 'acard_data', 'application.csv')
    df_data = pd.read_csv(fp_data, index_col=None)

    return df_data



def main():
    df_data = load_data()
    print(len(df_data))

    df_label = data_helper.Data.train_test_split(df_data, profile.pk, profile.label)
    print(len(df_label))

    df_feature = df_label[[profile.pk, profile.sample_type]].merge(df_data, on=profile.pk, how='left')
    data_helper.Data.dump('df_feature', df_feature, prefix=profile.prefix)
    print(len(df_feature))


if __name__ == '__main__':
    main()
