import cv2

# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text,clf):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id,_ = clf.predict(gray_img[y:y+h,x:x+w])
        # Check for id of user and label the rectangle accordingly
        if id==1:
            cv2.putText(img, "Abhi", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)                  
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

# Method to recognize the person
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img


# Method to detect the features
def detect(img, faceCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    if len(coords) == 4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        # Assign unique id to each user
        user_id = 1
        # img_id to make the name of each image unique
        generate_dataset(roi_img, user_id, img_id)

    return img

# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture=cv2.VideoCapture(0)
# Initialize img_id with 0
img_id = 0

while True:
    if img_id % 50 == 0:
        print("Collected ", img_id, " images")
    _, img = video_capture.read()
    # img = detect(img, faceCascade,img_id)
    img = recognize(img, clf, faceCascade)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
