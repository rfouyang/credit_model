import os
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import ROOT_DIR


class Dir:
    @classmethod
    def get_or_create(cls, fp_path):
        if isinstance(fp_path, str):
            fp_path = Path(fp_path)
        if Path.exists(fp_path):
            return fp_path
        else:
            os.mkdir(fp_path)
            return fp_path


class Data:
    fp_cache = Path(ROOT_DIR, "data")

    @classmethod
    def dump(cls, name, df, **kwargs):
        prefix = kwargs.get("prefix", None)
        if prefix is not None:
            fp_path = Dir.get_or_create(Path(cls.fp_cache, prefix))
        else:
            fp_path = Dir.get_or_create(cls.fp_cache)

        with open(Path(fp_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(df, f)

    @classmethod
    def load(cls, name, **kwargs):
        prefix = kwargs.get("prefix", None)
        if prefix is not None:
            fp_path = Path(cls.fp_cache, prefix)
        else:
            fp_path = cls.fp_cache

        with open(Path(fp_path, f"{name}.pkl"), "rb") as f:
            df = pickle.load(f)
            return df

    @classmethod
    def train_test_split(cls, df_sample, pk, label, **kwargs):
        test_size = kwargs.get("test_size", 0.2)
        random_state = kwargs.get("random_state", 20180105)
        reserved = kwargs.get("reserved", list())

        df_label = df_sample[[pk, label] + reserved].copy()
        df_train, df_test = train_test_split(
            df_label,
            test_size=test_size,
            stratify=df_sample[label],
            random_state=random_state,
        )

        idx_train = df_label[pk].isin(df_train[pk].tolist())
        idx_test = df_label[pk].isin(df_test[pk].tolist())

        df_label.loc[:, "sample_type"] = "unknown"
        df_label.loc[idx_train, "sample_type"] = "train"
        df_label.loc[idx_test, "sample_type"] = "test"

        return df_label

    @classmethod
    def combine_sample(cls, pk, df_label, lst_ft):
        df_res = df_label
        for df in lst_ft:
            df_res = df_res.merge(df, on=pk, how="left")
        return df_res

    @classmethod
    def save_gif(cls, fp_dir, fp_dst):
        import glob

        import imageio

        images = list()
        for f in sorted(glob.glob(f"{str(fp_dir)}/*.png")):
            print(f)
            images.append(imageio.imread_v2(f))
        imageio.mimsave(str(fp_dst), images)
