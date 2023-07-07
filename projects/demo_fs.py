from pathlib import Path
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toad

from util import woe_helper
from util import woe_mono
from util import stable_selection

from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score


def main():
    fp_base = Path('/Users', 'rfouyang', 'workspace')
    # fp_base = Path('C:\\', 'Users', 'USER', 'workspace')
    fp_data = Path(fp_base, 'data', 'data_woe_result.csv')

    df_woe = pd.read_csv(fp_data, index_col=None)

    num_cols = ['Collateral_valuation', 'Age', 'Properties_Total', 'Amount', 'Term', 'Historic_Loans', 'Current_Loans',
                'Max_Arrears']
    cat_cols = ['Region', 'Area', 'Activity', 'Guarantor', 'Collateral', 'Properties_Status']
    features = num_cols + cat_cols
    label = 'Defaulter'

    #score = MRMR.mrmr(df_woe[features].values, df_woe[label].values)
    df_woe = df_woe.sample(5000)
    score = fisher_score.fisher_score(df_woe[features].values, df_woe[label].values, mode='index')
    print(score)

if __name__ == '__main__':
    main()