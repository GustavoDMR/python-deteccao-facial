import cv2  # biblioteca para manipulação e reconhecimento de imagens

# chama código de IA previamente treinado que reconhece faces

classificador = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

# imagem escolhida / tratamento para preto e branco

imagem = cv2.imread('pessoas\\eu2.jpeg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# parametros de escala, número minimio de vizinhos e tamanho minimo de rosto

facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))

print(len(facesDetectadas))  # coordenadas em pixels dos rostos
print(facesDetectadas)

# loop para destacar pixels em vermelho, criando retangulo em volta do rosto

for (x, y, l, a) in facesDetectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow("Faces encontradas ", imagem)
cv2.waitKey()
