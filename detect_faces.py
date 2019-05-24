from imutils import paths, resize
import face_recognition
import argparse
import pickle
import cv2
import os
import time
import tensorflow as tf
from align import detect_face
import numpy as np

def find_faces(imagePaths, module='face_recognition', detection_method='cnn'):
    if module == 'face_recognition':
        print("[INFO] started with {} method {}".format(module, detection_method))
    elif module == 'opencv':
        print("[INFO] started with {}".format(module))
    elif module == 'facenet':
        print("[INFO] started with {}".format(module))
        sess = tf.Session()
        # read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

    # initialize the list of known encodings and known names
    knownBoxes = []
    knownNames = []

    if module == 'opencv':
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-1]

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        if module == 'face_recognition':
            image_recolor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif module == 'opencv':
            image_recolor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif module == 'facenet':
            image_recolor = image

        boxes = []
        if module == 'face_recognition':
            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(image_recolor, model=detection_method)
        elif module == 'opencv':
            boxes = face_cascade.detectMultiScale(image_recolor, 1.1, 5)
        elif module == 'facenet':
            minsize = 20
            threshold = [0.6, 0.7, 0.7]
            factor = 0.709
            margin = 44

            def getFace(img):
                bxs = []
                img_size = np.asarray(img.shape)[0:2]
                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                if not len(bounding_boxes) == 0:
                    for face in bounding_boxes:
                        if face[4] > 0.50:
                            det = np.squeeze(face[0:4])
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - margin / 2, 0)
                            bb[1] = np.maximum(det[1] - margin / 2, 0)
                            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                            bxs.append(([bb[0], bb[1], bb[2], bb[3]]))
                return bxs
            boxes = getFace(image_recolor)

        # loop over the boxes
        # for encoding in encodings:
        for box in boxes:
            # add each encoding + name to our set of known names and
            # encodings
            knownBoxes.append(box)
            knownNames.append(name)
    return {"boxes": knownBoxes, "names": knownNames}

def draw_image_information(image, name, data, label, color):
    boxes = [data['boxes'][i] for i, v in enumerate(data['names']) if v == name]
    for (top, right, bottom, left) in boxes:
        if label == "facenet":
            cv2.rectangle(image, (top, right), (bottom, left), color, 2)
            y = bottom - 15 if bottom - 15 > 15 else bottom + 15
            cv2.putText(image, label, (right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        else:
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return image

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", type=str, required=True, default="dataset",
                    help="path to input directory of images")
    ap.add_argument("-o", "--output", type=str, required=True, default="output",
                    help="path to output directory of images")
    ap.add_argument("-m", "--module", type=str, default="face_recognition",
                    help="face detection module to use: either `face_recognition` or `cv` or 'facenet'")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                    help="face detection model to use: either `hog` or `cnn`")
    args = vars(ap.parse_args())

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(args["dataset"]))

    if args["module"] == "all":
        start_time = time.time()
        result_fn = find_faces(imagePaths, "facenet")
        print("[INFO] --- %s seconds --- " % (time.time() - start_time))

        start_time = time.time()
        result_cv = find_faces(imagePaths, "opencv")
        print("[INFO] --- %s seconds --- " % (time.time() - start_time))

        start_time = time.time()
        result_fr_hog = find_faces(imagePaths, "face_recognition", "hog")
        print("[INFO] --- %s seconds --- " % (time.time() - start_time))

        #start_time = time.time()
        #result_fr_cnn = find_faces(imagePaths, "face_recognition", "cnn")
        #print("[INFO] --- %s seconds --- " % (time.time() - start_time))
    else:
        start_time = time.time()
        result = find_faces(imagePaths, args["module"], args["detection_method"])
        print("[INFO] --- %s seconds --- " % (time.time() - start_time))

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-1]
        image = cv2.imread(imagePath)

        if args["module"] == "all":
            draw_image_information(image, name, result_fn, "facenet", (0, 255, 0))
            draw_image_information(image, name, result_cv, "cv", (0, 0, 255))
            draw_image_information(image, name, result_fr_hog, "hog", (255, 0, 0))
            #draw_image_information(image, name, result_fr_cnn, "cnn", (255, 0, 255))
        else:
            draw_image_information(image, name, result, args["module"], (0, 255, 0))

        print("[INFO] saving modified image {} ".format(name))
        cv2.imwrite(args["output"] + os.path.sep + name, image);