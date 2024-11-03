from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model


# Train the model
results = model.train(data='/home/dslab/Documents/Rooftop detection yolov8/data.yaml', epochs=25)
