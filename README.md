# AI Video Filter

AI Video Filter is a Python application that leverages webcam input to generate AI-enhanced images in real time using the Stable Diffusion model. Users can select their camera, set style transfer parameters, and capture images with artistic modifications based on their prompts.

## Features

- **Camera Selection:** Choose from available webcams.
- **Real-time Video Feed:** Live view from your selected camera.
- **AI Image Generation:** Generate AI images based on real-time input.
- **Prompt Input:** Customize the prompt to influence AI-generated images.
- **Style Transfer Strength:** Control the intensity of the style transfer effect.
- **Take Picture:** Capture an image and apply the AI filter.

## Installation

To run this project, follow the steps below:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anttiluode/AiVideoFilter.git
   cd AI-Video-Filter
Create a virtual environment (optional but recommended):

python -m venv env

source env/bin/activate  # On Windows use `env\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

Usage

Run the application:

python app.py

Select the desired camera from the dropdown menu and click "Open Camera". (Usually 0)

Enter a prompt in the provided text field (e.g., "A portrait of a person with pipe") and adjust the style transfer strength using the slider.
The weaker the slider the more like original, the stronger, the more like the prompt. 

Click "Start Live Generation" to begin processing the video feed. Use the "Take Picture" button to capture a still image and generate an AI-enhanced version.
Currently it is making images with stable-diffuion 2.1 but for example Claude AI can easily change that for you to another model. 

Currently there is a time.sleep(5) at the app.py so that there is 5 second pause between the rendered images but if you have 
faster computer than mine (3060 ti + 5500 ryzen) you can easily make that way faster. 

Click "Close" to exit the application.

Requirements
The application uses the following libraries:

OpenCV
NumPy
PyTorch
Diffusers
PIL (Pillow)
Tkinter
License
This project is licensed under the MIT License.
