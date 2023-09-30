import modules.scripts as scripts
import gradio as gr
import os

from cuda import cudart
from modules import images, script_callbacks, sd_unet, paths_internal, devices
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state, walk_files
from modules.ui_common import create_refresh_button

from utilities import Engine
import ui_trt

import torch
text_maxlen = 77
unet_dim = 4
embedding_dim = 2048
time_dim=6


def check_dims(batch_size, image_height, image_width):
    assert batch_size >= 1 and batch_size <= 4
    assert image_height % 8 == 0 or image_width % 8 == 0
    latent_height = image_height // 8
    latent_width = image_width // 8
    assert latent_height >= 64 and latent_height <= 256
    assert latent_width >= 64 and latent_width <= 256
    return (latent_height, latent_width)

def get_shape_dict(batch_size, image_height, image_width):
    latent_height, latent_width = check_dims(batch_size, image_height, image_width)
    return {
        'sample': (2 * batch_size, unet_dim, latent_height, latent_width),
        'encoder_hidden_states': (2 * batch_size, text_maxlen, embedding_dim),
        'latent': (2 * batch_size, 4, latent_height, latent_width),
        'text_embeds': (2 * batch_size, 1280),
        'time_ids': (2 * batch_size, time_dim),
    }

class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return TrtUnet(self.filename)
class TrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.engine = None
        self.shared_device_memory = None
        self.stream = None
        self.use_cuda_graph = False

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def allocate_buffers(self, batch_size, image_height, image_width):

        self.engine.allocate_buffers(shape_dict=get_shape_dict(batch_size, image_height, image_width),
                                                 device=devices.device)


    def infer(self, feed_dict):
        batch_size = int(feed_dict['x'].shape[0]/2)
        image_height = feed_dict['x'].shape[3]*8
        image_width = feed_dict['x'].shape[2]*8

        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (image_width, image_height)

        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=feed_dict['context'].dtype
        )
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(devices.device)

        params = {"sample": feed_dict['x'], "timestep": feed_dict['timesteps'].squeeze(0)[0], "encoder_hidden_states": feed_dict['context'], "text_embeds": feed_dict['y'], 'time_ids':add_time_ids}

        self.allocate_buffers(batch_size, image_width, image_height)
        noise_pred = self.engine.infer(params, self.stream, use_cuda_graph=self.use_cuda_graph)['latent']
        return noise_pred

    def forward(self, x, timesteps, context, y, *args, **kwargs):
        noise_pred = self.infer({"x": x, "timesteps": timesteps, "context": context[:,:154,:], "y":y[:,:1280]})

        return noise_pred.to(dtype=x.dtype, device=devices.device)

    def activate(self):
        self.engine = Engine(self.filename)
        self.engine.load()
        err, self.stream = cudart.cudaStreamCreate()
        max_device_memory = self.engine.engine.device_memory_size
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        self.engine.activate(reuse_device_memory=self.shared_device_memory)

    def deactivate(self):
        self.engine = None
        self.shared_device_memory = None
        devices.torch_gc()
        cudart.cudaStreamDestroy(self.stream)
        del self.stream
def list_unets(l):
    trt_dir = os.path.join(paths_internal.models_path, 'Unet-trt')
    candidates = list(walk_files(trt_dir, allowed_extensions=[".plan"]))
    for filename in sorted(candidates, key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]

        opt = TrtUnetOption(filename, name)
        l.append(opt)


script_callbacks.on_list_unets(list_unets)

script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)


