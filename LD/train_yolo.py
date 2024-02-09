from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('yolov8s.pt')

# Train
results = model.train(
   data='datasets/17R_dataset/dataset.yaml',
   name='yolov8s_R17_p300_scale0.3',
   imgsz=565,
   mosaic=0,
   scale=0.3,
   plots=True,
   save=True,
   epochs=1000,
   patience=300
)
