from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("image.jpg")

results = model(image)

annotated_image = results[0].plot()

cv2.imwrite("result.jpg", annotated_image)

cv2.imshow("AI Object Detection", annotated_image)

cv2.waitKey(0)

cv2.destroyAllWindows()