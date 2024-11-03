from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/dslab/Documents/Rooftop detection yolov8/data.yaml', epochs=25)