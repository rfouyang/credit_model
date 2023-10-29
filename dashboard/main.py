from pathlib import Path
import gradio as gr
from config import BASE_DIR
from dashboard.interfaces import data_loader
from dashboard.interfaces import woe_encoding

with open(Path(BASE_DIR, 'css', 'main.css'), 'r') as f:
    css = f.read()

theme = gr.themes.Default(
    font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea'
)


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
        title='Credit Engine',
        css=css,
        theme=theme
    )

    main_interface.launch()


def main():
    create_interface()


if __name__ == '__main__':
    main()
