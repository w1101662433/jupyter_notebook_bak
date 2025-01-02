from ultralytics import YOLO
model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')
results = model.train(data=r"D:\BaiduSyncdisk\jupyter\017_包包分类\dataset", epochs=30, val=False, auto_augment=None, workers=1)
res = model.predict(r"D:\BaiduSyncdisk\jupyter\017_包包分类\dataset\train\M41426\12543298-9bd81d46-72f2-4cab-a44d-7c931f9b5248.jpg",)
print(res)