from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg-pf.pt")

# Run prediction on webcam (source=0). No prompts required.
results = model.predict(0, show=True)