import gradio as gr

class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image and video Processing"
        self.ui_obj = ui_obj

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Face Analysis with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem("Playing with Webcam"):
                    with gr.Row():
                        webcam_image_in = gr.Image(label="Webcam Image Input", source="webcam")
                        webcam_video_in = gr.Video(label="Webcam Video Input", source="webcam")
    
    def launch_ui(self):
        self.ui_obj.launch()

if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()

    print("AI_FallingAsleepDriving")
