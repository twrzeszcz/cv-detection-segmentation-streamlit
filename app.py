import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, ClientSettings
import av
import cv2
import csv
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from PIL import Image

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

@st.cache(allow_output_mutation=True)
def train(df):
    df.dropna(axis=1, inplace=True)
    X = df.drop('class', axis=1)
    y = df['class']
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    pipeline.fit(X, y)
    del df, X, y

    return pipeline

def main_section():
    st.title('Simple Computer Vision Application')

def body_language_decoder():
    train_or_predict = st.sidebar.selectbox('Select type', ['Stream and Save', 'Stream, Train and Predict'])

    if train_or_predict == 'Stream and Save':
        stream_type = st.selectbox('Select streaming type', ['Stream only', 'Stream and save'])

        @st.cache
        def gen_feature_names():
            landmarks = ['class']
            for val in range(543):
                landmarks.extend(['x' + str(val), 'y' + str(val), 'z' + str(val), 'v' + str(val)])
            return landmarks

        landmarks = gen_feature_names()

        model_type = st.selectbox('Select type of the model', ['Pose and Face', 'Left Hand', 'Right Hand', 'Left and Right Hand'])

        if st.button('Generate New File'):
            with open('coords.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        class_name = st.text_input('Enter class name')


        class BodyDecoder(VideoProcessorBase):
            def __init__(self) -> None:
                self.class_name = None
                self.save = None
                self.model_type = None

            @st.cache
            def load_model_utils(self):
                mp_drawing = mp.solutions.drawing_utils
                mp_holistic = mp.solutions.holistic
                holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

                return mp_drawing, mp_holistic, holistic

            def live_stream(self, image):
                mp_drawing, mp_holistic, holistic = self.load_model_utils()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                          )

                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                          )

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                         )

                return image

            def live_stream_save(self, image):
                mp_drawing, mp_holistic, holistic = self.load_model_utils()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                          )

                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                          )

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                         )

                try:
                    if self.model_type == 'Pose and Face':
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())
                        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.face_landmarks.landmark]).flatten())
                        row = [self.class_name] + pose_row + face_row

                    elif self.model_type == 'Left Hand':
                        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.left_hand_landmarks.landmark]).flatten())
                        row = [self.class_name] + left_hand_row

                    elif self.model_type == 'Right Hand':
                        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.right_hand_landmarks.landmark]).flatten())
                        row = [self.class_name] + right_hand_row

                    elif self.model_type == 'Left and Right Hand':
                        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.left_hand_landmarks.landmark]).flatten())
                        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.right_hand_landmarks.landmark]).flatten())
                        row = [self.class_name] + left_hand_row + right_hand_row

                    with open('coords.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)

                except:
                    pass

                return image

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                image = frame.to_ndarray(format="bgr24")

                if self.save == 'Stream only':
                    image = self.live_stream(image)
                else:
                    image = self.live_stream_save(image)

                return av.VideoFrame.from_ndarray(image, format="bgr24")


        webrtc_ctx = webrtc_streamer(
            key="body_decoder",
            mode=WebRtcMode.SENDRECV,
            client_settings=WEBRTC_CLIENT_SETTINGS,
            video_processor_factory=BodyDecoder,
            async_processing=True
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.class_name = class_name
            webrtc_ctx.video_processor.save = stream_type
            webrtc_ctx.video_processor.model_type = model_type

    if train_or_predict == 'Stream, Train and Predict':
        model_type = st.selectbox('Select type of the model', ['Pose and Face', 'Left Hand', 'Right Hand', 'Left and Right Hand'])

        upload_file = st.file_uploader('Choose a csv file', type='csv')
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.success('Successfully uploaded')
            model = train(df)
            st.success('Successfully trained')

        class BodyPredictor(VideoProcessorBase):
            def __init__(self) -> None:
                self.model_type = None

            @st.cache
            def load_model_utils(self):
                mp_drawing = mp.solutions.drawing_utils
                mp_holistic = mp.solutions.holistic
                holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

                return mp_drawing, mp_holistic, holistic

            def live_stream(self, image):
                mp_drawing, mp_holistic, holistic = self.load_model_utils()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                          )

                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                          )

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                         )


                try:
                    if self.model_type == 'Pose and Face':
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())
                        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.face_landmarks.landmark]).flatten())
                        X = pd.DataFrame([pose_row + face_row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        del X, pose_row, face_row

                    elif self.model_type == 'Left Hand':
                        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.left_hand_landmarks.landmark]).flatten())
                        X = pd.DataFrame([left_hand_row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        del X, left_hand_row

                    elif self.model_type == 'Right Hand':
                        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.right_hand_landmarks.landmark]).flatten())
                        X = pd.DataFrame([right_hand_row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        del X, right_hand_row

                    elif self.model_type == 'Left and Right Hand':
                        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.left_hand_landmarks.landmark]).flatten())
                        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.right_hand_landmarks.landmark]).flatten())
                        X = pd.DataFrame([left_hand_row + right_hand_row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        del X, left_hand_row, right_hand_row

                    coords = tuple(np.multiply(np.array((
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640, 480]).astype(int))

                    cv2.rectangle(image, (coords[0], coords[1] + 5),
                                  (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                                  (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                    cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class, (90, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(np.max(body_language_prob)), (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                except:
                    pass

                return image


            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                image = frame.to_ndarray(format="bgr24")

                image = self.live_stream(image)

                return av.VideoFrame.from_ndarray(image, format="bgr24")


        webrtc_ctx = webrtc_streamer(
            key="body_predictor",
            mode=WebRtcMode.SENDRECV,
            client_settings=WEBRTC_CLIENT_SETTINGS,
            video_processor_factory=BodyPredictor,
            async_processing=True
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.model_type = model_type

def body_segmentation():
    img = st.file_uploader('Choose a image file', type=['jpg', 'png'])
    if img is not None:
        img = cv2.resize(np.array(Image.open(img)), (640, 480))
        st.image(img)
        st.success('Successfully uploaded')

    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)

    class BodySegmentation(VideoProcessorBase):
        def __init__(self) -> None:
            self.confidence_threshold = 0.5

        @st.cache(allow_output_mutation=True)
        def load_bodypix_model(self):
            bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
            return bodypix_model

        def live_stream(self, image):
            model = self.load_bodypix_model()

            result = model.predict_single(image)
            mask = result.get_mask(threshold=self.confidence_threshold).numpy().astype(np.uint8)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            inverse_mask = np.abs(result.get_mask(threshold=self.confidence_threshold).numpy() - 1).astype(np.uint8)
            masked_background = cv2.bitwise_and(img, img, mask=inverse_mask)
            final = cv2.add(masked_image, cv2.cvtColor(masked_background, cv2.COLOR_BGR2RGB))

            return final


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            image = self.live_stream(image)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="body_segmentation",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=BodySegmentation,
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold




activities = ['Main', 'Face Mask Detection', 'Body Language Decoder', 'Body Segmentation', 'About']
section_type = st.sidebar.selectbox('Select Option', activities)

if section_type == 'Main':
    main_section()

if section_type == 'Body Language Decoder':
    body_language_decoder()

if section_type == 'Body Segmentation':
    body_segmentation()









