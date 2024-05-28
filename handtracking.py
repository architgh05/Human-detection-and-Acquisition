from sre_constants import SUCCESS
import cv2 as cv2
import mediapipe as mp
import time
from numpy import True_

#to deetect the hand .

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands                          #hand tracking module
hands = mpHands.Hands()                            #object for the hands function.
mpDraw = mp.solutions.drawing_utils

pTime = 0 
cTime = 0

while True: 
    SUCCESS, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)            #converting it into rgb
    results = hands.process(imgRGB)

    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:                        #to check whether info received is of one hand or multiple hands.
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape                                     #height,width nd channels of our image.
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy) 
                if id == 4:
                    cv2.circle(img,(cx,cy),10,(255,25,255),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)             #drws the points and the connections.
    
    cTime=time.time()                                           #function that calcultes the current time.
    fps = 1 / (cTime-pTime)
    pTime = cTime

    
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
                      #fps in interger for , coordinates,font, scale,colour(purple),thickness.
    cv2.imshow("Image", img)
    cv2.waitKey(1) 
