from ultralytics import YOLO

# Load model
model = YOLO("./yolov8n-seg.pt")

# Export model
success = model.export(task="segment", format="onnx", opset=12, imgsz=640, simplify=True)