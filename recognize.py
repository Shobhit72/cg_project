# import datetime
# import os
# import time
# import cv2
# import pandas as pd


# def recognize_attendence():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()  
#     recognizer.read("TrainingImageLabel"+os.sep+"Trainer.yml")
#     harcascadePath = "haarcascade_default.xml"
#     faceCascade = cv2.CascadeClassifier(harcascadePath)
#     df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Id', 'Name', 'Date', 'Time']
#     attendance = pd.DataFrame(columns=col_names)

#     # start realtime video capture
#     cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     cam.set(3, 640) 
#     cam.set(4, 480) 
#     minW = 0.1 * cam.get(3)
#     minH = 0.1 * cam.get(4)

#     while True:
#         ret, im = cam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5,
#                 minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
#         for(x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
#             Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
#             if conf < 100:
#                 aa = df.loc[df['Id'] == Id]['Name'].values
#                 confstr = "  {0}%".format(round(100 - conf))
#                 tt = str(Id)+"-"+aa
#             else:
#                 Id = '  Unknown  '
#                 tt = str(Id)
#                 confstr = "  {0}%".format(round(100 - conf))

#             if (100-conf) > 67:
#                 ts = time.time()
#                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 aa = str(aa)[2:-2]
#                 attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

#             tt = str(tt)[2:-2]
#             if(100-conf) > 67:
#                 tt = tt + " [Pass]"
#                 cv2.putText(im, str(tt), (x+5,y-5), font, 1, (255, 255, 255), 2)
#             else:
#                 cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

#             if (100-conf) > 67:
#                 cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
#             elif (100-conf) > 50:
#                 cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
#             else:
#                 cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)


#         attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
#         cv2.imshow('Attendance', im)
#         if (cv2.waitKey(1) == ord('q')):
#             break
#     ts = time.time()
#     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#     Hour, Minute, Second = timeStamp.split(":")
#     fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
#     attendance.to_csv(fileName, index=False)
#     print("Attendance Successful")
#     cam.release()
#     cv2.destroyAllWindows()
import datetime
import os
import time
import cv2
import pandas as pd


def recognize_attendance():
    # Initialize recognizer and load trained model
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(os.path.join("TrainingImageLabel", "Trainer.yml"))
    except Exception as e:
        print(f"Error loading recognizer: {e}")
        return

    # Load Haarcascade file for face detection
    harcascadePath = "haarcascade_default.xml"
    if not os.path.exists(harcascadePath):
        print("Error: Haarcascade file not found")
        return

    faceCascade = cv2.CascadeClassifier(harcascadePath)
    
    # Load student details
    try:
        df = pd.read_csv(os.path.join("StudentDetails", "StudentDetails.csv"))
    except Exception as e:
        print(f"Error loading student details: {e}")
        return

    # Prepare attendance dataframe
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Start real-time video capture
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 100:
                name = df.loc[df['Id'] == Id]['Name'].values[0]
                confstr = "  {0}%".format(round(100 - conf))
                tt = f"{Id}-{name}"
                if (100 - conf) > 67:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]

                    tt += " [Pass]"
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
            else:
                tt = "Unknown"
                confstr = "  {0}%".format(round(100 - conf))
                color = (0, 0, 255)

            cv2.putText(im, tt, (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(im, confstr, (x+5, y+h-5), font, 1, color, 1)

        # Remove duplicate entries
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Attendance', im)
        if cv2.waitKey(1) == ord('q'):
            break

    # Save attendance record
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = os.path.join("Attendance", f"Attendance_{date}_{Hour}-{Minute}-{Second}.csv")
    attendance.to_csv(fileName, index=False)
    print("Attendance Successful")

    # Release resources
    cam.release()
    cv2.destroyAllWindows()


