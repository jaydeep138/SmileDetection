import cv2
trained_data_smile=cv2.CascadeClassifier('C:\jd\All\hello\AI\haarcascade_smile.xml')
trained_data_face = cv2.CascadeClassifier('C:\jd\All\hello\AI\haarcascade_frontalface_default.xml')
trained_data_eye=cv2.CascadeClassifier('C:\jd\All\hello\AI\haarcascade_eye.xml')
video_stream=cv2.VideoCapture(0)

while True:
    frame_read,frame=video_stream.read()

    if frame_read:
        grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        face_coordinates=trained_data_face.detectMultiScale(grayscaled_img)
        eye_coordinates=trained_data_eye.detectMultiScale(grayscaled_img,minNeighbors=20)

        for x,y,w,h in face_coordinates:

            face=frame[y:y+h , x:x+w]
            grayscaled_img2=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            smile=trained_data_smile.detectMultiScale(grayscaled_img2,scaleFactor=1.7,minNeighbors=20)

            if len(smile)>0:
                cv2.putText(frame,'smiling',(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        for x,y,w,h in eye_coordinates:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        
        cv2.imshow('JD car detector',frame)
        key=cv2.waitKey(1)
        if key==81 or key==113:
            break
    else:
        break


print("code completed")