from ultralytics import YOLO
#from wandb.integration.ultralytics import add_wandb_callback

# Load pretrained yolo model
model = YOLO('yolov8l.pt')
#model.cuda()
#add_wandb_callback(model, enable_model_checkpointing=True)

# Train
results = model.train(
   data='datasets/17R_dataset/dataset.yaml',
   name='yolov8l_R17_p300_aug_scale0.3',
   degrees=180,
   scale=0.3,
   fliplr=0.0,
   imgsz=576,
   mosaic = 0,
   perspective = 0.0001,
   plots=True,
   save=True,
   epochs=300
)


