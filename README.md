# E-Waste Component Detection

A Streamlit application that uses a trained Ultralytics YOLO model to detect e-waste components in uploaded images or frames from a webcam.

## Detected classes

The available `best.pt` model was inspected locally and contains these classes:

- `capacitor`
- `resistor`
- `ic_chip`
- `pcb`

Class names come from the trained model itself. If you replace the model with another training run, verify its class names before relying on this list.

## Current model requirement

`app.py` currently loads the model at startup with:

```python
model = YOLO(r"X:\\best.pt")
```

That is an absolute path to a Windows `X:` drive. The app will fail during startup if that drive is unavailable or if `X:\\best.pt` does not exist. A local `best.pt` in this repository folder does not satisfy the current code because it is a different path.

To connect the model stored beside `app.py`, change that line to:

```python
model = YOLO("best.pt")
```

Alternatively, copy the model to `X:\\best.pt`, or update the line to the absolute path where the file is stored. Keep the model path consistent with the machine running Streamlit.

The repository's `.gitignore` excludes `best.pt`, so the local model is intentionally not committed. The app cannot perform detections without a compatible trained model file. It can only start after the model path resolves to a valid YOLO checkpoint.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Obtain the trained `best.pt` checkpoint and make it available at the path used by `app.py`.

   Do not commit large model files directly to Git. For local-only use, place `best.pt` beside `app.py` after changing the path as shown above. For collaboration, use Git LFS or publish the checkpoint as a release/external artifact, such as a Hugging Face Hub model repository, and download it during setup. Keep credentials and private model URLs out of source control.

## Run

From the project directory, start Streamlit:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

Use the sidebar to set the confidence threshold and choose between:

- **Image upload:** upload a `.jpg`, `.jpeg`, or `.png` file.
- **Webcam:** enable webcam mode and allow camera access. OpenCV must be able to access camera device `0`.

## Project files

- `app.py` - Streamlit user interface and YOLO inference code.
- `best.pt` - local trained model checkpoint; ignored by Git and not included in the repository.
- `sample_images/` - sample images for manual testing.
- `requirements.txt` - Python packages imported by the application.
