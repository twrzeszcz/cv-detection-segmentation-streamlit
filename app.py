import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, ClientSettings
import av
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from PIL import Image
from utils import visualize_boxes_and_labels_on_image_array
import gc


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

@st.cache(allow_output_mutation=True)
def train(saved_landmarks, landmarks):
    df = pd.DataFrame(saved_landmarks, columns=landmarks[:len(saved_landmarks[0])])
    X = df.drop('class', axis=1)
    y = df['class']
    del df
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    pipeline.fit(X, y)
    del X, y

    return pipeline

def main_section():
    st.title('Simple Computer Vision Application')
    st.image('main_background.jpg')
    st.markdown('This application has 3 features which utilize different machine learning models. The first app is a body '
                'language detector where the *mediapipe* package is used to detect face, pose and hands landmarks. User can choose '
                'one of these and capture corresponding landmarks to train a model. More details are explained in a **Body Language Decoder** '
                'section.')
    st.markdown('In the **Body Segmentation** section we are using the *tf_bodypix* package for segmentation. User can upload an image '
                'which is then used as a background.')
    st.markdown('The last feature is a **Face Mask Detector** where the object detection model trained with a *Tensorflow Object Detection API* '
                'is used. More details about this feature can be found here [Github](https://github.com/twrzeszcz/face-mask-detection-streamlit).')
    st.markdown('First and second feature were implemented according to the tutorials on [YouTube1](https://www.youtube.com/watch?v=We1uB79Ci-w&t=2690s) '
                'and [YouTube2](https://www.youtube.com/watch?v=0tB6jG55mig&t=317s).')

def body_language_decoder():
    train_or_predict = st.sidebar.selectbox('Select type', ['Stream and Save', 'Stream, Train and Predict'])

    @st.cache(allow_output_mutation=True)
    def get_data():
        saved_landmarks = []
        return saved_landmarks

    saved_landmarks = get_data()

    @st.cache
    def gen_feature_names():
        landmarks = ['class']
        for val in range(543):
            landmarks.extend(['x' + str(val), 'y' + str(val), 'z' + str(val), 'v' + str(val)])
        return landmarks

    landmarks = gen_feature_names()

    if train_or_predict == 'Stream and Save':
        st.markdown('There are 2 types of streaming that you can choose here. It is either just the live webcam stream with displayed landmarks or a '
                    'live stream when the selected landmarks are saved. You can choose from 4 different types of landmarks to save: *Pose and Face*, '
                    '*Left Hand*, *Right Hand*, *Left and Right Hand*. To save landmarks you have to also specify the **class name** so the name of the '
                    'eg. expression, gesture etc. To stop the live stream and saving just press the **Stop** button. To get the new class from the same type '
                    'of landmarks you have to update the **class name** and start stream again. There is currently no option to have landmarks from different '
                    'types in the same file. To use a different type you can press **Clear saved landmarks**.')

        stream_type = st.selectbox('Select streaming type', ['Stream only', 'Stream and save'])

        model_type = st.selectbox('Select type of the model', ['Pose and Face', 'Left Hand', 'Right Hand', 'Left and Right Hand'])

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

                    saved_landmarks.append(row)


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
        st.markdown('In this section a simple machine learning model is trained on the saved landmarks. You only have to '
                    'select the type of landmarks that were saved before.')

        model_type = st.selectbox('Select type of the model', ['Pose and Face', 'Left Hand', 'Right Hand', 'Left and Right Hand'])

        model = train(saved_landmarks, landmarks)
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

                    img_shape = list(image.shape[:-1])
                    img_shape.reverse()

                    coords = tuple(np.multiply(np.array((
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), img_shape).astype(int))

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


    if st.button('Clear saved landmarks'):
        saved_landmarks.clear()

    st.write('Total number of saved landmarks: ' + str(len(saved_landmarks)))

def body_segmentation():
    img = st.file_uploader('Choose a image file', type=['jpg', 'png'])
    if img is not None:
        img = np.array(Image.open(img))
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

            img_shape = list(image.shape[:-1])
            img_shape.reverse()
            image_shape = tuple(img_shape)

            inverse_mask = np.abs(result.get_mask(threshold=self.confidence_threshold).numpy() - 1).astype(np.uint8)
            masked_background = cv2.bitwise_and(cv2.resize(img, image_shape), cv2.resize(img, image_shape), mask=inverse_mask)
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

def face_mask_detection():
    @st.cache
    def load_model():
        detect_fn = tf.saved_model.load('my_model_mobnet/saved_model')

        return detect_fn

    detect_fn = load_model()


    class MaskDetector(VideoProcessorBase):
        def __init__(self) -> None:
            self.confidence_threshold = 0.5
            self.category_index = {1: {'id': 1, "name": 'with_mask'}, 2: {'id': 2, 'name': 'without_mask'},
                                    3: {'id': 3, 'name': 'mask_weared_incorrect'}}
            self.num_boxes = 1

        def gen_pred(self, image):
            input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            visualize_boxes_and_labels_on_image_array(
                image,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=self.num_boxes,
                min_score_thresh=self.confidence_threshold,
                agnostic_mode=False)

            return image

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            image = self.gen_pred(image)

            return av.VideoFrame.from_ndarray(image, format="bgr24")



    webrtc_ctx = webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=MaskDetector,
        async_processing=True,
    )

    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    num_boxes = st.slider('Number of boxes', 1, 20, 1)

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
        webrtc_ctx.video_processor.num_boxes = num_boxes

activities = ['Main', 'Body Language Decoder', 'Body Segmentation', 'Face Mask Detector']
section_type = st.sidebar.selectbox('Select Option', activities)

if section_type == 'Main':
    main_section()

if section_type == 'Body Language Decoder':
    body_language_decoder()
    gc.collect()

if section_type == 'Body Segmentation':
    body_segmentation()
    gc.collect()

if section_type == 'Face Mask Detector':
    face_mask_detection()
    gc.collect()









