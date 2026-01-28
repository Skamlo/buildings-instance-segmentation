import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
dotenv_path = os.path.join(parent_dir, '.env')
import gradio as gr
from modules.model.model import InstanceSegmentation
import torch
from modules.utils.visualization import visualize_prediction
from dotenv import load_dotenv
load_dotenv(dotenv_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InstanceSegmentation()
checkpoint_path = os.getenv("PATH_TO_MODEL")
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.to(device)
model.eval()


def process_manual_upload(image):
    if image is None:
        print("Error: No image uploaded")
        return None
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)    
    return visualize_prediction(img_tensor[0], prediction[0])

    
with gr.Blocks(title="Image Processing") as demo:
    gr.Markdown("## Choose Image Source")    

    # with gr.Tabs():
        # with gr.Tab("Download from Geoportal"):
        #     gr.Markdown("Enter coordinates.")
        #     with gr.Row():
        #         input_lat = gr.Number(label="Latitude", value=55.75)
        #         input_lon = gr.Number(label="Longitude", value=37.61)
        #         input_zoom = gr.Slider(1, 20, value=15, step=1, label="Zoom Level")
        #     btn_geoportal = gr.Button("Download", variant="primary")

    with gr.Tab("Upload File"):
        gr.Markdown("Upload your photo.")
        input_image_file = gr.Image(label="Source Image", type="numpy")
        btn_upload = gr.Button("Process", variant="primary")

    output_image = gr.Image(label="Result", type="numpy", interactive=False)

    # btn_geoportal.click(
    #     fn=process_from_geoportal,
    #     inputs=[input_lat, input_lon, input_zoom],
    #     outputs=[output_image]
    # )

    btn_upload.click(
        fn=process_manual_upload,
        inputs=[input_image_file],
        outputs=[output_image]
    )

if __name__ == "__main__":
    demo.launch()