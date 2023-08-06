import gradio as gr
from dashboard.interfaces import data_loader
from dashboard.interfaces import woe_encoding


def create_interface():
    # Define the data loader tab
    data_loader_tab = data_loader.UI.create_ui()
    woe_encoding_tab = woe_encoding.UI.create_ui()

    # Create the tabbed interface
    main_interface = gr.TabbedInterface(
        interface_list=[
            data_loader_tab,
            woe_encoding_tab
        ],
        tab_names=['Data', 'WOE'],
        title='Credit Engine'
    )

    main_interface.launch()


def main():
    create_interface()


if __name__ == '__main__':
    main()
