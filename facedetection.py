import cv2
import predict
from keras.models import load_model
classes = ['Jayesh','Kanan','Shaalin','Vaandana']
cascPath ="C:/Users/Shalin/Anaconda3/Library/etc/haarcascades/"
faceCascade = cv2.CascadeClassifier(cascPath + 'haarcascade_frontalface_default.xml' )
model = load_model('my_model.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_face(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        _val, img = cam.read()
        if mirror:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = faceCascade.detectMultiScale(gray_img,
							 scaleFactor = 1.2,
							 minNeighbors = 5)

            for (x, y, w, h) in face:
                result = predict.predict_face(img, model)
                name = classes[result[0].tolist().index(max(result[0]))]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, name, (x,y-5), font, .6,(0,255,0),1,cv2.LINE_AA)
            cv2.imshow('frame',img)
            if cv2.waitKey(5) & 0xFF == ord('q'):       #quit program on pressing 'Q'
	             break
    cam.release()
    cv2.destroyAllWindows()  

#    img = img[face[0][1]-75:face[0][1]+face[0][3]+75,face[0][0]-50:face[0][0]+face[0][2]+50]
#    cv2.imwrite('face.png',img)
    
def main():
    detect_face(mirror=True)

if __name__ == '__main__':
	main()