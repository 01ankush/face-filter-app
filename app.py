# Importing required libraries, obviously
from io import BytesIO, BufferedReader

import dlib
import numpy as np
import streamlit as st
from PIL import Image
import cv2
# only for google colab, not required if running in on local system
import matplotlib
import os
import imutils
from imutils import face_utils

# Loading pre-trained parameters for the cascade classifier
try:
    # facial landmarks predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

except Exception:
    st.write("Error loading shape_predictor_68_face_landmarks")

def distance(a, b):
    # a -> (x1, y1)
    # b -> (x2, y2)
    # used to calculate distance between two points
    dis = np.sqrt( (b[0] - a[0])**2 + (b[1] - a[1])**2 )
    return int(dis)

def get_image_gray(image):
    # reading image
    image = cv2.imread(image)
    # resizing for better performance
    image = imutils.resize(image, width=512)
    # converting to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect(image):
    '''
    Function to detect faces/eyes and smiles in the image passed to this function
    '''

    nose_filter_main = cv2.imread('nose.png')
    tongue_filter_main = cv2.imread('tongue.png')
    detector = dlib.get_frontal_face_detector()
    image, gray = get_image_gray(image)

    faces = detector(gray, 1)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_np = face_utils.shape_to_np(landmarks)

        # face orientation calculation
        nose_line = landmarks_np[27:31]
        delta_x = landmarks_np[27, 0] - landmarks_np[30, 0]
        delta_y = landmarks_np[27, 1] - landmarks_np[30, 1]
        face_orientation = np.arctan2(delta_y, delta_x) * 180 / np.pi
        if face_orientation < 0:
            face_orientation = face_orientation + 90
        else:
            face_orientation = face_orientation - 90

        # Nose
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        # center of nose -> mean of all nose points
        nose_anchor = np.mean(landmarks_np[30:36], axis=0)
        # calculating nose width
        nose_width = int(distance(left_nose, right_nose) * 2)  # scaling factor -> 2
        # rotating filter according to face orientation
        nose = imutils.rotate_bound(nose_filter_main.copy(), face_orientation)
        # resizing filter according to nose width
        nose = imutils.resize(nose, width=nose_width)

        # Nose ROI
        rows_nose, cols_nose, _ = nose.shape
        nose_height = rows_nose
        top_left_nose = (int(nose_anchor[0] - nose_width / 2), int(nose_anchor[1] - nose_height / 2))
        bottom_right_nose = (int(nose_anchor[0] + nose_width / 2), int(nose_anchor[1] + nose_height / 2))
        nose_roi = image[top_left_nose[1]: top_left_nose[1] + nose_height,
                   top_left_nose[0]: top_left_nose[0] + nose_width]

        # creating masks for nose
        nose_gray = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_gray, 10, 255, cv2.THRESH_BINARY)
        nose_mask_inv = cv2.bitwise_not(nose_mask)

        # Now black-out the area of filter in ROI
        background_nose = cv2.bitwise_and(nose_roi, nose_roi, mask=nose_mask_inv)
        # Take only region of filter from filter image.
        foreground_nose = cv2.bitwise_and(nose, nose, mask=nose_mask)

        # merging background_nose and foreground_nose
        dst = cv2.add(background_nose, foreground_nose)
        image[top_left_nose[1]: top_left_nose[1] + nose_height, top_left_nose[0]: top_left_nose[0] + nose_width] = dst

        # Mouth
        left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
        # center of mouth -> mean of all (49, 55, 8, 10) points
        tongue_anchor = np.mean(landmarks_np[np.r_[48, 54, 7, 9]], axis=0)
        # calculating mouth width
        tongue_width = int(distance(left_mouth, right_mouth) * 1)  # scaling factor -> 1.5
        # rotating filter according to face orientation
        tongue = imutils.rotate_bound(tongue_filter_main.copy(), face_orientation)
        # resizing filter according to nose width
        tongue = imutils.resize(tongue, width=tongue_width)

        # Tongue ROI
        rows_tongue, cols_tongue, _ = tongue.shape
        tongue_height = rows_tongue
        # top_left_tongue = (int(tongue_anchor[0] - tongue_width / 2), int(tongue_anchor[1] - tongue_height / 2))
        # bottom_right_tongue = (int(tongue_anchor[0] + tongue_width / 2), int(tongue_anchor[1] + tongue_height / 2))
        top_left_tongue = left_mouth
        bottom_right_tongue = (int(left_mouth[0] + tongue_width), int(left_mouth[1] + tongue_height))
        tongue_roi = image[top_left_tongue[1]: top_left_tongue[1] + tongue_height,
                     top_left_tongue[0]: top_left_tongue[0] + tongue_width]

        # creating masks for nose
        tongue_gray = cv2.cvtColor(tongue, cv2.COLOR_BGR2GRAY)
        _, tongue_mask = cv2.threshold(tongue_gray, 10, 255, cv2.THRESH_BINARY)
        tongue_mask_inv = cv2.bitwise_not(tongue_mask)

        # Now black-out the area of filter in ROI
        background_tongue = cv2.bitwise_and(tongue_roi, tongue_roi, mask=tongue_mask_inv)
        # Take only region of filter from filter image.
        foreground_tongue = cv2.bitwise_and(tongue, tongue, mask=tongue_mask)

        # merging background_nose and foreground_nose
        dst = cv2.add(background_tongue, foreground_tongue)
        image[top_left_tongue[1]: top_left_tongue[1] + tongue_height,
        top_left_tongue[0]: top_left_tongue[0] + tongue_width] = dst
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Returning the image with bounding boxes drawn on it (in case of detected objects), and faces array
    return image


def about():
    st.write(
        '''
        Heyy mates! 
Here you can give a funky look to your boring pictures by applying cute filters on them. 
For now, we have a dog face filter where you just have to click a picture of yours through webcam
 and the filter will put the nose and tongue of a dog to your face. The best part of the filter is that it can be applied to all the faces captured in the photograph. Besides this, you can also download the filtered image so that it will remain with you as a amusing memory.
So what are you waiting for! Just start giving your selfies as well as group images a funky touch and create fun memories.
        ''')


def main():
    st.title("Instagram filter App :sunglasses: ")
    st.write("**Using the shape_predictor_68_face_landmarks**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Explore more", activities)

    if choice == "Home":

        st.write("Go to the About section from the sidebar to learn more about it.")

        # You can specify more file types below if you want
        # image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        # picture = st.camera_input("Take a picture")
        #
        # if picture:
        #     st.image(picture)
        picture = st.camera_input("First, take a picture...")

        if picture:
            with open('test.jpg', 'wb') as file:
                file.write(picture.getbuffer())



        if picture is not None:

            # image = Image.open(picture)

            if st.button("Process"):
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img = detect("test.jpg")
                st.image(result_img, use_column_width=True,channels="RGB")
                # st.success("Found {} faces\n".format(len(result_faces)))
                im_rgb = result_img[:, :, [2, 1, 0]]  # numpy.ndarray
                ret, img_enco = cv2.imencode(".png", im_rgb)  # numpy.ndarray
                srt_enco = img_enco.tostring()  # bytes
                img_BytesIO = BytesIO(srt_enco)  # _io.BytesIO
                img_BufferedReader = BufferedReader(img_BytesIO)  # _io.BufferedReader

                st.download_button(
                    label="Download",
                    data=img_BufferedReader,
                    file_name="filtered_img.png",
                    mime="image/png"
                )

        #  st.download_button(
        #     label="Download image",
        #     data=file,
        #     file_name=result_img,
        #     mime="image/png"
        # )


    elif choice == "About":
        about()


if __name__ == "__main__":
    main()