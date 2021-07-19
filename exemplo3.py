import cv2 # biblioteca para manipulação e reconhecimento de imagens

video = cv2.VideoCapture(0) # variavel que captura o video usando método VideoCapture


# método classificador de faces

classificadorFace = cv2.CascadeClassifier('cascade\\haarcascade_frontalface_default.xml')

# loop para coleta de video

while True:
    conectado, frame = video.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize = (70, 70))

    # for que cria um retangulo vermelho nas dimensoes do rosto

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    # método waitKey para finalizar coleta de video

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


