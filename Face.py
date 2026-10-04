import FacialCam

Emotion_code = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

password = [6,4,5]
attempt = []
for i in range(len(password)):
    emotion = FacialCam.capture_video()
    print(emotion)
    attempt.append(Emotion_code.get(emotion))



print("Password: " + str(password) + "\nAttempt: " + str(attempt))

print(attempt == password)