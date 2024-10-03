import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tkinter import Tk, Label, Button, Entry, StringVar, Scale, HORIZONTAL, OptionMenu
from threading import Thread
import time

class AIVideoFilter:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Video Filter")

        # Initialize video capture
        self.cap = None
        self.camera_index = 0  # Default camera index
        self.available_cameras = self.get_camera_indices()

        # Create a dropdown for camera selection
        self.camera_var = StringVar(master)
        self.camera_var.set(self.available_cameras[0])  # Default value
        self.camera_menu = OptionMenu(master, self.camera_var, *self.available_cameras)
        self.camera_menu.pack()

        self.open_camera_button = Button(master, text="Open Camera", command=self.open_camera)
        self.open_camera_button.pack()

        # Create a label for video feed
        self.video_label = Label(master)
        self.video_label.pack()

        # Create an entry for the prompt
        self.prompt_var = StringVar()
        self.prompt_entry = Entry(master, textvariable=self.prompt_var, width=50)
        self.prompt_entry.pack()
        self.prompt_entry.insert(0, "A portrait of a person with ")

        # Create a slider for style transfer strength
        self.strength_var = StringVar()
        self.strength_scale = Scale(master, from_=0.1, to=1.0, resolution=0.1, orient=HORIZONTAL,
                                    label="Style Transfer Strength", variable=self.strength_var)
        self.strength_scale.set(0.75)  # Default value
        self.strength_scale.pack()

        # Create buttons to control the application
        self.start_live_button = Button(master, text="Start Live Generation", command=self.start_live_generation)
        self.start_live_button.pack()

        self.take_picture_button = Button(master, text="Take Picture", command=self.take_picture)
        self.take_picture_button.pack()

        self.close_button = Button(master, text="Close", command=self.close)
        self.close_button.pack()

        # Store the current frame
        self.current_frame = None
        self.live_generation = False

        # Load the Stable Diffusion model
        print("Loading Stable Diffusion model...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        print("Model loaded!")

        # Start the video feed
        self.update_video()

    def get_camera_indices(self):
        """Get the list of available camera indices."""
        indices = []
        for i in range(5):  # Check first 5 device indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                indices.append(f"Camera {i}")
                cap.release()
        return indices

    def open_camera(self):
        """Open the selected camera."""
        selected_camera = self.camera_var.get()
        self.camera_index = int(selected_camera.split(" ")[-1])  # Get the camera index
        if self.cap is not None:
            self.cap.release()  # Release any previously opened camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

    def update_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()  # Store the current frame for image generation
                # Convert the BGR frame to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Display the real image in the window
                cv2.imshow("Webcam Feed", rgb_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

        self.master.after(10, self.update_video)

    def start_live_generation(self):
        if not self.live_generation:
            self.live_generation = True
            self.start_live_button.config(text="Stop Live Generation")
            self.image_generation_thread = Thread(target=self.live_generate_ai_image)
            self.image_generation_thread.start()
        else:
            self.live_generation = False
            self.start_live_button.config(text="Start Live Generation")

    def live_generate_ai_image(self):
        while self.live_generation:
            if self.current_frame is not None:
                # Convert the current frame to PIL Image
                image = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                # Resize the image to 512x512 (or another size that works well with your model)
                image = image.resize((512, 512))

                # Get the strength value from the slider
                strength = float(self.strength_var.get())

                # Generate the image
                result = self.pipe(
                    prompt=self.prompt_var.get(),
                    image=image,
                    strength=strength,  # Use the strength from the slider
                    guidance_scale=7.5
                ).images[0]

                # Display the result
                result.show()

                # Sleep for a short while to avoid overwhelming the system
                time.sleep(5)  # Adjust this time as necessary (e.g., 5 seconds)

    def take_picture(self):
        if self.current_frame is not None:
            # Convert the current frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
            # Resize the image to 512x512
            image = image.resize((512, 512))

            # Get the strength value from the slider
            strength = float(self.strength_var.get())

            # Generate the AI image
            result = self.pipe(
                prompt=self.prompt_var.get(),
                image=image,
                strength=strength,
                guidance_scale=7.5
            ).images[0]

            # Display the result
            result.show()

    def close(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.master.quit()

if __name__ == "__main__":
    root = Tk()
    app = AIVideoFilter(root)
    root.mainloop()
