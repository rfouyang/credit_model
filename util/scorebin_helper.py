import numpy as np
import toad


class ScoreBin:
    @classmethod
    def bin_quantile(cls, df_score, score, label, **kwargs):
        n_bins = kwargs.get("n_bins", 10)
        bin_name = kwargs.get("bin_name", "bin")
        combiner = toad.transform.Combiner()

        combiner.fit(
            df_score[[score, label]],
            y=label,
            method="quantile",
            n_bins=n_bins,
            empty_separate=True,
        )

        df_score[bin_name] = combiner.transform(df_score[[score]], labels=True)
        return df_score

    @classmethod
    def bin_step(cls, df_score, score, label, **kwargs):
        n_bins = kwargs.get("n_bins", 10)
        bin_name = kwargs.get("bin_name", "bin")
        combiner = toad.transform.Combiner()

        combiner.fit(
            df_score[[score, label]],
            y=label,
            method="step",
            n_bins=n_bins,
            empty_separate=True,
        )

        df_score[bin_name] = combiner.transform(df_score[[score]], labels=True)
        return df_score

    @classmethod
    def bin_given(cls, df_score, score, splits, **kwargs):
        bin_name = kwargs.get("bin_name", "bin")
        combiner = toad.transform.Combiner()

        combiner.update({score: splits})

        df_score[bin_name] = combiner.transform(df_score[[score]], labels=True)
        return df_score

    @classmethod
    def bin_subscore(cls, df_score, score, **kwargs):
        splits = [0, 30] + list(np.arange(50, 90, 5)) + [100]
        bin_name = kwargs.get("bin_name", "bin")
        combiner = toad.transform.Combiner()

        combiner.update({score: splits})

        df_score[bin_name] = combiner.transform(df_score[[score]], labels=True)
        return df_score

    @classmethod
    def bin_megascore(cls, df_score, score, **kwargs):
        splits = [0, 300] + list(np.arange(500, 900, 50)) + [1000]
        bin_name = kwargs.get("bin_name", "bin")
        combiner = toad.transform.Combiner()

        combiner.update({score: splits})

        df_score[bin_name] = combiner.transform(df_score[[score]], labels=True)
        return df_score
