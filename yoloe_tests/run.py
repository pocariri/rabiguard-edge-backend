from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg-pf.pt")

# Run prediction. No prompts required.
results = model.predict("./image.jpg")

# Show results
results[0].show()