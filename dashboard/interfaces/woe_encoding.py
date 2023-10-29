import gradio as gr
import pandas as pd
from pprint import pprint
import pickle
from pathlib import Path
from config import DATA_DIR
from util import woe_helper, report_helper

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

        df_data = data['data']
        label = data['label']

        Data.data = df_data
        Data.features = data['features']
        Data.label = data['label']

        df_stat = df_data.agg(total=(label, 'count'),
                              bad=(label, 'sum'),
                              bad_rate=(label, 'mean'))

        return df_stat.reset_index()

    @classmethod
    def listner_button_woe(cls, method):
        woe = woe_helper.WOE()
        woe.fit(Data.data[Data.features + [Data.label]], Data.label, method=method)

        fp_woe = Path(DATA_DIR, 'woe.pkl')
        with open(fp_woe, 'wb') as f:
            pickle.dump(woe, f)

        df_bin = woe.transform(Data.data[Data.features + [Data.label]], bin_only=True)
        report = report_helper.FTReport.get_report(df_bin, Data.features, Data.label)

        fp_report = Path(DATA_DIR, 'report.xlsx')
        report.to_excel(fp_report)

        return fp_woe, fp_report

    @classmethod
    def create_ui(cls):
        with gr.Blocks() as block:
            with gr.Row():
                loader = gr.inputs.components.File(type='file', label='Dataset in PKL format')
                btn_load = gr.Button("load")
            with gr.Row():
                stat = gr.outputs.components.Dataframe(type='pandas', label='Stat')

            with gr.Row():
                btn_woe = gr.Button("WOE")
                slt_method = gr.inputs.components.Dropdown(choices=['dt', 'chi'], value='dt')
            with gr.Row():
                model = gr.outputs.components.File(label='model')
                report = gr.outputs.components.File(label='report')

            btn_load.click(cls.listner_button_load, inputs=loader, outputs=stat)

            btn_woe.click(cls.listner_button_woe, inputs=slt_method, outputs=[model, report])

            return block


def main():
    UI.create_ui().launch()


if __name__ == '__main__':
    main()
