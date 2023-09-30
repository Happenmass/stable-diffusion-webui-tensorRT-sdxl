import html
import os

import launch
from modules import script_callbacks, paths_internal, shared, devices
import gradio as gr

from modules.shared import cmd_opts
from modules.ui_components import FormRow

import tensorrt as trt

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
import pycuda.driver as cuda


from utilities import Engine

text_maxlen = 77
unet_dim = 4
embedding_dim = 2048
time_dim=6


def getEnginePath(model_name):
    return os.path.join(paths_internal.models_path, 'Unet-trt', model_name+ '.plan')

def getonnxPath(model_name):
    return os.path.join(paths_internal.models_path, 'Unet-onnx', model_name + '.onnx')
def get_input_profile(batch_size, min_bs, max_bs, min_width, max_width, min_height, max_height, static_batch):
    image_height = 1024
    image_width = 1024
    latent_height = image_height // 8
    latent_width = image_width // 8
    min_latent_height = latent_height//8 if static_batch else min_height //8
    min_latent_width = latent_width//8 if static_batch else min_width //8
    max_latent_height = latent_height//8 if static_batch else max_height //8
    max_latent_width = latent_width //8 if static_batch else max_width //8
    return {
        'sample': [(2 * min_bs, unet_dim, min_latent_height, min_latent_width),
                   (2 * max_bs, unet_dim, latent_height, latent_width),
                   (2 * max_bs, unet_dim, max_latent_height, max_latent_width)],
        'encoder_hidden_states': [(2 * min_bs, text_maxlen, embedding_dim),
                                  (2 * max_bs, text_maxlen, embedding_dim),
                                  (2 * max_bs, text_maxlen, embedding_dim)],
        'text_embeds': [(2 * min_bs, 1280), (2 * max_bs, 1280), (2 * max_bs, 1280)],
        'time_ids': [(2 * min_bs, time_dim), (2 * max_bs, time_dim), (2 * max_bs, time_dim)]
    }

def convert_onnx_to_trt(trt_filename, trt_source_filename, min_bs, max_bs, min_width, max_width, min_height, max_height,use_fp16, *args):
    # 初始化CUDA驱动
    # cuda.init()
    # cuda.Device(0).make_context()

    if not os.path.exists(os.path.join(paths_internal.models_path, 'Unet-trt')):
        os.makedirs(os.path.join(paths_internal.models_path, 'Unet-trt'))

    profile = get_input_profile(1,  min_bs, max_bs, min_width, max_width, min_height, max_height, False)
    p = Profile()
    for name, dims in profile.items():
        assert len(dims) == 3
        p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    config_kwargs = {}
    config_kwargs['tactic_sources'] = []

    network = network_from_onnx_path(getonnxPath(trt_source_filename), flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
    engine = engine_from_network(
        network,
        config=CreateConfig(fp16=use_fp16,
                            refittable=False,
                            profiles=[p],
                            load_timing_cache=None,
                            **config_kwargs
                            ),
        save_timing_cache=None
    )
    save_engine(engine, path=getEnginePath(trt_filename))

    del network
    del engine

    return "tensorrt model generated"



def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="Convert OPT ONNX to TensorRT"):
                        trt_source_filename = gr.Textbox(label='Opt Onnx model filename', value="", elem_id="trt_source_filename")
                        trt_filename = gr.Textbox(label='Output filename', value="", elem_id="trt_filename", info="Leave empty to use the same name as onnx and put results into models/Unet-trt directory")

                        with gr.Column(elem_id="trt_width"):
                            min_width = gr.Slider(minimum=768, maximum=2048, step=64, label="Minimum width", value=1024, elem_id="trt_min_width")
                            max_width = gr.Slider(minimum=768, maximum=2048, step=64, label="Maximum width", value=1024, elem_id="trt_max_width")

                        with gr.Column(elem_id="trt_height"):
                            min_height = gr.Slider(minimum=768, maximum=2048, step=64, label="Minimum height", value=1024, elem_id="trt_min_height")
                            max_height = gr.Slider(minimum=768, maximum=2048, step=64, label="Maximum height", value=1024, elem_id="trt_max_height")

                        with gr.Column(elem_id="trt_batch_size"):
                            min_bs = gr.Slider(minimum=1, maximum=4, step=1, label="Minimum batch size", value=1, elem_id="trt_min_bs")
                            max_bs = gr.Slider(minimum=4, maximum=4, step=1, label="Maximum batch size", value=4, elem_id="trt_max_bs")



                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            use_fp16 = gr.Checkbox(label='Use half floats', value=True, elem_id="trt_fp16")

                        button_export_trt = gr.Button(value="Convert ONNX to TensorRT", variant='primary', elem_id="trt_convert_from_onnx")

            with gr.Column(variant='panel'):
                trt_result = gr.Label(elem_id="trt_result", value="", show_label=False)
                trt_info = gr.HTML(elem_id="trt_info", value="")

        button_export_trt.click(
            convert_onnx_to_trt,
            inputs=[trt_filename, trt_source_filename, min_bs, max_bs, min_width, max_width, min_height, max_height,use_fp16],
            outputs=[trt_result, trt_info],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]

