import gradio as gr
import pandas as pd
from pprint import pprint
import pickle
from pathlib import Path
from config import DATA_DIR


class Data:
    data = None


class UI:

    @classmethod
    def listner_button_load(cls, file):
        fp_path = file.name
        if fp_path.endswith('.pkl'):
            with open(fp_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = None

        Data.data = data
        print(data)
        df_data = data['data']
        label = data['label']

        df_stat = df_data.agg(total=(label, 'count'),
                              bad=(label, 'sum'),
                              bad_rate=(label, 'mean'))

        return df_stat.reset_index()

    @classmethod
    def create_ui(cls):
        with gr.Blocks() as block:
            with gr.Row():
                loader = gr.inputs.components.File(type='file', label='Dataset in PKL format')
                btn_load = gr.Button("load")
            with gr.Row():
                stat = gr.outputs.components.Dataframe(type='pandas', label='Stat')
            btn_load.click(cls.listner_button_load, inputs=loader, outputs=stat)

            return block


def main():
    UI.create_ui().launch()


if __name__ == '__main__':
    main()
