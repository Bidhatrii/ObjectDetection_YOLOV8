import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="YOLOv8 live")
  parser.add_argument(
      "--webcam-resolution",
      default=[640, 480],  # Use a lower resolution for better processing like 640, 480 rather than 1280, 720
      nargs=2,
      type=int
  )
  args = parser.parse_args()
  return args

def main():
  args = parse_arguments()
  frame_width, frame_height = args.webcam_resolution

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

  model = YOLO("yolov8n.pt")

  box_annotator = sv.BoxAnnotator(
      thickness=2,
      text_thickness=2,
      text_scale=1
  )

  while True:
      ret, frame = cap.read()

      if not ret:
          break

      result = model(frame)[0]
      detections = sv.Detections.from_yolov8(result)

      labels = [
          f"{model.model.names[class_id]} {confidence:0.2f}"
          for  _, confidence, class_id, _
          in detections
      ]

      frame = box_annotator.annotate(
          scene=frame, 
          detections=detections, 
          labels=labels
          )

      cv2.imshow("yolov8", frame)

      if cv2.waitKey(1) == 27:  # Wait for 1ms, if 'Esc' is pressed (ASCII 27), exit
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()

