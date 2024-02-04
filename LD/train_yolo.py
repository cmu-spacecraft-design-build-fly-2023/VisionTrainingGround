from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('yolov8l.pt')

# Train
results = model.train(
   data='datasets/toy_dataset_new/dataset.yaml',
   imgsz=1536,
   epochs=200,
   batch=2,

   # Augmentation Parameters
   degrees=0.0,
   scale=0.1,
   fliplr=0.0,
   translate=0.0, 
   shear=0.0,
   mosaic = 0,
   perspective = 0,
   hsv_h=0.055,
   name='yolov8l_R17',
   plots=True,
   save=True
)
