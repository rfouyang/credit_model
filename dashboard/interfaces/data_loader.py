import gradio as gr
import pandas as pd
from pprint import pprint
import pickle
from pathlib import Path
from config import DATA_DIR


class Param:
    df_data = pd.DataFrame()
    lst_col = list()
    pk = None
    exclude = list()
    label = None


class UI:

    @classmethod
    def listner_button_load(cls, file):
        fp_path = file.name
        if fp_path.endswith('.xlsx'):
            Param.df_data = pd.read_excel(fp_path)
        elif fp_path.endswith('.csv'):
            Param.df_data = pd.read_csv(fp_path)
        else:
            Param.df_data = pd.DataFrame()

        Param.lst_col = list(Param.df_data.columns)

        options = Param.lst_col
        slt_pk = gr.inputs.Dropdown.update(choices=options, value=options[0])
        slt_label = gr.inputs.Dropdown.update(choices=options, value=options[0])
        slt_exclude = gr.inputs.Dropdown.update(choices=options, value=[])
        return Param.df_data.head(), slt_pk, slt_label, slt_exclude

    @classmethod
    def listner_button_save(cls):
        param = {
            'data': Param.df_data,
            'lst_col': Param.lst_col,
            'pk': Param.pk,
            'label': Param.label,
            'exclude': Param.exclude,
            'features': sorted(set(Param.lst_col) - {Param.pk, Param.label} - set(Param.exclude))
        }

        fp_save = Path(DATA_DIR, 'data.pkl')
        with open(fp_save, "wb") as f:
            pickle.dump(param, f)

        return fp_save

    @classmethod
    def set_pk(cls, value):
        Param.pk = value
        return value

    @classmethod
    def set_label(cls, value):
        Param.label = value
        return value

    @classmethod
    def set_exclude(cls, value):
        Param.exclude = value
        return value

    @classmethod
    def create_ui(cls):
        with gr.Blocks() as block:
            # loading data
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        loader = gr.inputs.components.File(type='file', label='Dataset')
                    with gr.Row():
                        btn_load = gr.Button("load", variant='primary', size='sm')

                with gr.Column():
                    frame = gr.outputs.components.Dataframe(type='pandas', interactive=True, label="Data Table")

            # set columns
            with gr.Row():
                with gr.Column():
                    slt_pk = gr.inputs.components.Dropdown(choices=list(), label="Primary Ley")
                    slt_label = gr.inputs.components.Dropdown(choices=list(), label="Label")
                    slt_exclude = gr.inputs.components.Dropdown(choices=list(), label="Exclude", multiselect=True)
                    btn_save = gr.Button("Save", variant='primary', size='sm')

                with gr.Column():
                    output = gr.outputs.components.File(label='Dataset in PKL Format')

            btn_load.click(cls.listner_button_load, inputs=loader,
                           outputs=[frame, slt_pk, slt_label, slt_exclude])
            btn_save.click(cls.listner_button_save, None, outputs=output)

            slt_pk.change(cls.set_pk, slt_pk, Param.pk)
            slt_label.change(cls.set_label, slt_label, Param.label)
            slt_exclude.change(cls.set_exclude, slt_exclude, Param.exclude)



            return block


def main():
    UI.create_ui().launch()


if __name__ == '__main__':
    main()
