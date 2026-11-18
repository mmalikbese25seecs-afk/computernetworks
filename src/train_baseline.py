"""
train_baseline.py
Trains the BASELINE detector (YOLOv8) on the SeaDronesSee subset.
This detector sits at the receiver (ground station) and looks for victims in the video it receives.
"""
from ultralytics import YOLO


def run(data_yaml, project_dir, model_name='yolov8s.pt', epochs=12, imgsz=960, batch=16, resume=False):
    if resume:
        model = YOLO(f'{project_dir}/baseline/weights/last.pt')       # continue an interrupted run
        return model.train(resume=True)
    model = YOLO(model_name)                                          # start from COCO-pretrained weights
    return model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                       project=project_dir, name='baseline', exist_ok=True,
                       workers=2, seed=0, plots=True)
