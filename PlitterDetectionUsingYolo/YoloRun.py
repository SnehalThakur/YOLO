from ultralytics import YOLO

model = YOLO(r"litterDetectionCustomModelV8.pt")
model.predict(r"C:\Users\snehal\PycharmProjects\PlitterDetectionUsingYolo\datasets\test\images\IMG_20220618_122620_jpg.rf.94a9bdf9280de1913b11c0b42f9d6a25.jpg", show=True, save=True, hide_labels=False, hide_conf=False, save_txt=True, conf=0.5, line_thickness=3)
# model.train(data="datasets/data.yaml", epochs=50)


# from roboflow import Roboflow
# rf = Roboflow(api_key="IaWy30TbFdzY7d4uL4e5")
# project = rf.workspace().project("merged-lrbku")
# model = project.version(1).model

# infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict(r"C:\Users\snehal\PycharmProjects\PlitterDetectionUsingYolo\can.jpg", confidence=40, overlap=30).save("prediction1.jpg")

# version.export("yolov8")
# version.download("yolov8")
# model = version.model

# model.save

# infer on a local image
# print(model.predict(r"C:\Users\snehal\PycharmProjects\PlitterDetectionUsingYolo\bottle.jpeg", confidence=40, overlap=30).json())



# yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=25 imgsz=640 plots=True batch=8