import cv2
import tensorflow as tf
import numpy as np
from typing import List
from . import label_map_util
import json

class TensorflowImageDetector:
  def __init__(self, pathToModel, pathToLabelMap):
    self.__detectionModel = tf.saved_model.load(pathToModel)
    self.__categoryIndex = label_map_util.create_category_index_from_labelmap(pathToLabelMap, use_display_name=True)

  def getDetectionBoundingBoxes(self, image: List[List[List[int]]], threshold: int, maxDetections: int):
    detections = self.__detectFromImage(image, self.__detectionModel)
    scores = detections['detection_scores'][0, :maxDetections].numpy()
    bboxes = detections['detection_boxes'][0, :maxDetections].numpy()
    labels = detections['detection_classes'][0, :maxDetections].numpy().astype(np.int64)
    labels = [self.__categoryIndex[n]['name'] for n in labels]
    (h, w, d) = image.shape
    detectedObjects = []
    for bbox, label, score in zip(bboxes, labels, scores):
      if score > threshold:
        xMin, yMin = int(bbox[1]*w), int(bbox[0]*h)
        xMax, yMax = int(bbox[3]*w), int(bbox[2]*h)
        topLeftPoint = [xMin, yMin]
        bottomRightPoint = [xMax, yMax]
        detectedObject = {
            "label": label,
            "topLeftPoint": topLeftPoint,
            "bottomRightPoint": bottomRightPoint,
            "score": score
        }
        detectedObjects.append(detectedObject)
    return detectedObjects

  def __detectFromImage(self, image, model):
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
    return model(input_tensor)

  def drawDetectionsOnImage(self, image, detections, allowedLabels = None):
    editedImage = image.copy()
    if isinstance(allowedLabels, list):
      for detection in detections:
        if detection['label'] in allowedLabels:
          self.__drawSingleDetection(editedImage, detection)
    else:
      for detection in detections:
        self.__drawSingleDetection(editedImage, detection)
    return editedImage

  def __drawSingleDetection(self, image, detection):
    cv2.rectangle(image, detection["topLeftPoint"], detection["bottomRightPoint"], (0,255,0), 2)
    cv2.putText(image, f"{detection['label']}: {int(detection['score']*100)} %", detection["topLeftPoint"], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

  def saveDetections(self, detections, filePath):
    convertedDetections = [{**detection, 'score': float(detection["score"])} for detection in detections]
    detectionsJson = json.dumps(convertedDetections)
    with open(filePath, 'w') as file:
      file.write(detectionsJson)
  
  def loadDetections(self, filePath):
    readFile = ""
    with open(filePath, 'r') as file:
      readFile = file.read()
    loadedDetections = json.loads(readFile)
    return loadedDetections