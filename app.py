import argparse
import cv2
import numpy as np
from yolo import YOLO
# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("Real Time Detection and Identification of American Sign Language")

print("loading yolo...")
yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}

def get_labels_for_plot(predictions):
    predictions_labels = []
    for i in range(len(predictions)):
        for ins in labels_dict:
            if predictions[i] == labels_dict[ins]:
                predictions_labels.append(ins)
                break
    return predictions_labels

# load model
print("Loading recognition model...")
model = load_model('model.h5')
# summarize model.
print("model loaded")
print(model.summary())

class VideoProcessor:
    def __init__ (self) -> None:
        self.threshold1 = 100
        self.threshold2 = 200

    def _annotate_image(self, frame, width, height, inference_time, results):
        # display fps
        cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)

        # sort by confidence
        results.sort(key=lambda x: x[2])

        # how many hands should be shown
        hand_count = len(results)

        # display hands
        for detection in results[:hand_count]:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            img_ = frame.astype(np.uint8)
            crop_img = img_[y-40 : y+h+40, x-40 : x+w+40]
            if crop_img.size != 0 :
                crop_image = cv2.resize(crop_img, (64, 64))
                crop_image = np.array(crop_image)
                crop_image = crop_image.astype('float32')/255.0
                res = model.predict(crop_image.reshape(1,64,64,3))
                classes_x = np.argmax(res,axis=1)
                class_label = get_labels_for_plot([classes_x])
                text = "%s (%s)" % (class_label, round(confidence, 2))
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        return frame, results

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        print("starting webcam...")
        image = frame.to_ndarray(format="bgr24")
        width, height, inference_time, results = yolo.inference(image)
        annotated_image, results = self._annotate_image(image, width, height, inference_time, results)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

webrtc_ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
