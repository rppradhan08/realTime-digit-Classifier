import cv2
from keras.models import load_model
import numpy as np

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65  # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
brightness = 100
#####################################

# CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)
cap.set(10, brightness)
# LOAD THE TRAINNED MODEL
# To load the model tensorflow_2.1.0 and keras_2.3.1 need to be installed
model = load_model('my_model.h5')

# PREPORCESSING FUNCTION


def preProcessing(img):
    '''
    Performs preprocessing on the raw image
    '''
    # converts BGR to GRAYSCALE image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # increases the contrast by equalizing the histogram
    img = cv2.equalizeHist(img)
    # normalize
    img = img/255
    img = cv2.resize(img, (28, 28))
    return img


# Loop to display continuous frames
while True:
    _, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = preProcessing(img)
    img = img.reshape(1, 28, 28, 1)

    classIndex = int(model.predict_classes(img))
    # Probability of predicted class
    probVal = np.amax(model.predict(img))
    print(classIndex, probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "   "+str(probVal),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
