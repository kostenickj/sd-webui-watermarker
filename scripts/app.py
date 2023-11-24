import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import StableDiffusionProcessing, process_images, Processed
from modules.processing import Processed
from modules.shared import opts
from modules import script_callbacks
from PIL import Image

class WaterMarkerScript(scripts.Script):  

    img2img_batch_output_dir:str = None
    def title(self):
        return "Watermarker"

    def show(self, is_img2img):
        return True
    
    @staticmethod
    def on_batch_dir_change(value:str, x, y):
        WaterMarkerScript.img2img_batch_output_dir = value
        return value
    
    @staticmethod
    def on_after_component_callback(component, **_kwargs):
        if component.elem_id == 'img2img_batch_output_dir':
            with gr.Blocks():
                c: gr.Textbox = component
                c.blur(fn= WaterMarkerScript.on_batch_dir_change, inputs=[component])

    def ui(self, is_img2img):
        with gr.Blocks():
            logo = gr.Image(label="Watermark Source", show_label=True, source="upload", interactive=True, type="pil",image_mode="RGBA", elem_id="watermarker_logo")
            overwrite = gr.Checkbox(False, label="Overwrite", elem_id="watermarker_logo_overwrite")
            alpha = gr.Slider(value=0.5, maximum=1, minimum=0.01, step=0.01 ,interactive=True ,label="Transparency", elem_id="watermarker_logo_alpha")
            scale = gr.Slider(value=0.5, maximum=1, minimum=0.1, step=0.01 ,interactive=True ,label="Scale", elem_id="watermarker_logo_scale")
            position = gr.Dropdown(choices=["Top Left", "Top Right", "Bottom Left", 'Bottom right', 'Center'], label="Position", value="Bottom Left", elem_id="watermarker_logo_position")
            return [logo, overwrite, alpha, position, scale]
    
    def run(self, p: StableDiffusionProcessing, _original_logo: Image, overwrite: bool, alpha: float, position: str, scale: float):
        def add_wm(im: Image, padding = 0):

            logo: Image = _original_logo.copy()

            height, width = im.size
            logo_height, logo_width = logo.size
            shorter_side = min(height, width)
            new_logo_width = int(shorter_side * scale)
            logo_aspect_ratio = logo.width / logo.height
            new_logo_height = int(new_logo_width / logo_aspect_ratio)

            is_smaller = new_logo_width < logo_width or new_logo_height < logo_height

            logo = logo.resize((new_logo_width, new_logo_height), resample=Image.Resampling.NEAREST if is_smaller else Image.Resampling.BICUBIC)
            pos_x, pos_y = 0, 0

            if position == 'Top Left':
                pos_x, pos_y = padding, padding
            elif position == 'Top Right':
                pos_x, pos_y = height - new_logo_width - padding, padding
            elif position == 'Bottom Left':
                pos_x, pos_y = padding, width - new_logo_height - padding
            elif position == 'Bottom right':
                pos_x, pos_y = height - new_logo_width - padding, width - new_logo_height - padding
            elif position == 'Center':
                pos_x, pos_y = (height - new_logo_width) // 2, (width - new_logo_height) // 2

            with_wm: Image = im.copy().convert('RGBA')
            alpha_channel = logo.getchannel('A')
            a_value = alpha * 255
            new_alpha = alpha_channel.point(lambda i: a_value if i > 0 else 0)
            logo.putalpha(new_alpha)
            with_wm.alpha_composite(logo, (pos_x, pos_y))
            return with_wm

        if not _original_logo:
            print('no watermark found, cancelling')
            proc = process_images(p)
            return proc

        if(overwrite):
            p.do_not_save_samples = True

        proc = process_images(p)
        print(WaterMarkerScript.img2img_batch_output_dir)

        
        print('adding watermark')
        # dewwit
        for i in range(len(proc.images)):
            proc.images[i] = add_wm(proc.images[i])
            images.save_image(
                proc.images[i], 
                p.outpath_samples, 
                proc.seed + i, 
                proc.prompt, 
                opts.samples_format, 
                info= proc.info, 
                p=p, 
                suffix="" if overwrite else "_WM"
            )

        return proc
    

script_callbacks.on_after_component(WaterMarkerScript.on_after_component_callback)