import cv2
import os
import pickle
import numpy as np
import face_recognition
import time
import requests
import sys
#------------
import dlib
from scipy.spatial import distance as dist
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
blink_counter = 0
ear_threshold = 0.3
def calculate_ear(eye_landmarks):
    # Compute the Euclidean distances between the vertical eye landmarks
    a = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    b = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    c = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Calculate the EAR
    ear = (a + b) / (2.0 * c)
    return ear

# Create a function to detect eye blinks
def detect_blinks(frame):
    print("Function Called")
    global blink_counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)

        # Extract the coordinates of the left and right eye landmarks
        left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate the EAR for each eye
        left_ear = calculate_ear(left_eye_landmarks)
        right_ear = calculate_ear(right_eye_landmarks)

        # Check for a blink
        if left_ear < ear_threshold and right_ear < ear_threshold:
            blink_counter += 1
            cv2.putText(frame, 'Human', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame
#-----------------------
# import cvzone
'''
adharNo = sys.argv[1]
d = {'adharNo': adharNo}
print("adhar" + " : " + adharNo)

post = requests.post('http://localhost:8001/faceRecoginiton/datapostCv', data=d)
data = post.json()
print("Response from server:", data)
img = data.get('photo')
print(img)
'''
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 425)  # Width
cap.set(4, 320)  # Height

# Added
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Display",cv2.WND_PROP_TOPMOST)
cv2.moveWindow("Display", 600, 150)

# ima=response.content
# cv2.imshow("Imsges",ima)
# cv2.waitKey(0)
# BackgroundImage=cv2.imread("./mode/Background.png")

'''
#Importing modes into list
ModeFolder='./Resources/mode'
modePathList=os.listdir(ModeFolder)
imgModeList=[] #List contining mode path
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(ModeFolder,path)))

    '''
'''
#Importing Images
Folder='Images'
PathList=os.listdir(Folder)
imgList=[] #List contining mode path
ID=[]


for path in PathList:
    imgList.append(cv2.imread(os.path.join(Folder,path)))
    ID.append(os.path.splitext(path)[0])
#print(ID)

'''
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

image_urls = ["https://avatars.githubusercontent.com/u/106619187?v=4"]
imgList = []  # List containing image data
ID = []
known_face_detected = False

'''
def posting():
    obj = {"result": known_face_detected, 'adharNo': adharNo}
    post = requests.post('http://localhost:8001/faceRecoginiton/datapostresultCv', data=obj)
    print("Post")
'''

for url in image_urls:
    # Download the image from the URL
    response = requests.get(url)  # Getting images
    img_array = np.frombuffer(response.content, dtype=np.uint8)  # Converting into numpy array
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Changinh into image
    # cv2.imshow("Fetched",img)
    # cv2.waitKey(1)
    imgList.append(img)
    ID.append(os.path.splitext(os.path.basename(url))[0])


def Encoding(imagesList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convering to RGB
        encode = face_recognition.face_encodings(img)[0]  # Getting first element of encoding
        encodeList.append(encode)
    return encodeList


print("Encoding Started..")

encodeListKnown = Encoding(imgList)
encodeListKnownIds = [encodeListKnown, ID]

print("Encoding Ends")

'''
file=open("Efile.p","wb")
pickle.dump(encodeListKnownIds,file)
file.close()
print("File Saved")


print("Loading Encoded File...")
file=open("Efile.p","rb")
encodeListKnownId = pickle.load(file)
'''

encodeListKnown, ID = encodeListKnownIds
# print("Encoded file Loaded..")
initime = time.time()
while True:

    time_limit = False
    _, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing to shorten computation
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting to RGB

    faceCurrFrame = face_recognition.face_locations(imgS)  #
    encodedCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    # BackgroundImage[154:154+480,70:70+640]=img
    # BackgroundImage[154:154 + 394, 830:830 + 307] = imgModeList[0]

    # Loop through each encodings

    for encodeFace, faceLoc in zip(encodedCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.40)
        faceDistance = face_recognition.face_distance(encodeListKnown,
                                                      encodeFace)  # lower the face distance better the match
        print(faceDistance)
        # print("matches",matches)
        # print("Face Distance",faceDistance)

        matchIndex = np.argmin(faceDistance)
        # print("Match INdex",matchIndex)

        if matches[matchIndex] and faceDistance <= 0.40:
            print("Known Face Detected")
            img = detect_blinks(img)
            if blink_counter>0:
                print("Inside")
                known_face_detected = True
                print(ID[matchIndex])
        if known_face_detected:
            break
            # print(ID[matchIndex])
            # y1,x2,y2,x1=faceLoc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # bbox = 70 + x1, 154 + y1,x2 - x1,  y2 - y1
            # BackgroundImage=cvzone.cornerRect(BackgroundImage,bbox,rt=0)

        fintime = time.time()
        if (fintime - initime >= 40):
            time_limit = True
            print("Time Limit Exceeded")
            break
    # post = requests.get('http://localhost:8001/faceRecoginiton/datapostresultCv',known_face_detected )

    if len(faceCurrFrame) > 0:
        face_locations = face_recognition.face_locations(img)
        for face_location in face_locations:
            (top, right, bottom, left) = face_location
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Display", img)

    # cv2.imshow("WebCam",img)
    # obj={"result":known_face_detected}
    # post = requests.post('http://localhost:8001/faceRecoginiton/datapostresultCv',data=obj )
    cv2.waitKey(1)
    if known_face_detected or time_limit:
        time.sleep(5)
        cap.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows

        break
#posting()
