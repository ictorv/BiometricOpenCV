import cv2
import os
import pickle
import numpy as np
import face_recognition
import time
import requests
import cvzone

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,640) #Width
cap.set(4,480) #Height

BackgroundImage=cv2.imread("Resources/Background.png")

#Importing modes into list
ModeFolder='Resources/mode'
modePathList=os.listdir(ModeFolder)
imgModeList=[] #List contining mode path
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(ModeFolder,path)))

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
image_urls=['https://media.licdn.com/dms/image/D5603AQH7I_3kdCXXrw/profile-displayphoto-shrink_800_800/0/1685678390121?e=1699488000&v=beta&t=Aa9vfE3QuBMZ3vYLZ1Bf2gQQ9B1IJKJU1R-MyFa4szQ']
imgList = []  # List containing image data
ID = []

for url in image_urls:
    # Download the image from the URL
    response = requests.get(url) #Getting images
    img_array = np.frombuffer(response.content, dtype=np.uint8) #Converting into numpy array
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) #Changinh into image

    imgList.append(img)
    ID.append(os.path.splitext(os.path.basename(url))[0])

def Encoding(imagesList):
    encodeList=[]
    for img in imgList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convering to RGB
        encode=face_recognition.face_encodings(img)[0] #Getting first element of encoding
        encodeList.append(encode)
    return encodeList

print("Encoding Started..")
encodeListKnown=Encoding(imgList)
encodeListKnownIds=[encodeListKnown,ID]

print("Encoding Ends")

file=open("EncodeFile.p","wb")
pickle.dump(encodeListKnownIds,file)
file.close()
print("File Saved")


print("Loading Encoded File...")
file=open("EncodeFile.p","rb")
encodeListKnownId = pickle.load(file)
encodeListKnown,ID=encodeListKnownId
print("Encoded file Loaded..")
initime=time.time()
while True:
    known_face_detected = False
    time_limit=False
    _,img=cap.read()

    imgS=cv2.resize(img,(0,0),None,0.25,0.25) #resizing to shorten computation
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converting to RGB

    faceCurrFrame=face_recognition.face_locations(imgS)  #
    encodedCurrFrame=face_recognition.face_encodings(imgS,faceCurrFrame)


    BackgroundImage[154:154+480,70:70+640]=img
    BackgroundImage[154:154 + 394, 830:830 + 307] = imgModeList[0]


    #Loop through each encodings

    for encodeFace, faceLoc in zip(encodedCurrFrame,faceCurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.40)
        faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace) #lower the face distance better the match
        print(faceDistance)
        #print("matches",matches)
        #print("Face Distance",faceDistance)

        matchIndex=np.argmin(faceDistance)
        #print("Match INdex",matchIndex)

        if matches[matchIndex] and faceDistance<=0.40:
            print("Known Face Detected")
            known_face_detected = True

            print(ID[matchIndex])
            break
            #print(ID[matchIndex])
            #y1,x2,y2,x1=faceLoc
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            #bbox = 70 + x1, 154 + y1,x2 - x1,  y2 - y1
            #BackgroundImage=cvzone.cornerRect(BackgroundImage,bbox,rt=0)

        fintime=time.time()
        if (fintime - initime >= 40):
            time_limit=True
            print("Time Limit Exceeded")
            break

    cv2.imshow("Display", BackgroundImage)
    #cv2.imshow("WebCam",img)

    cv2.waitKey(1)
    if known_face_detected or time_limit:
        time.sleep(5)
        cv2.destroyAllWindows()  # Close all OpenCV windows
        break
