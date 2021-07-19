import cv2 # biblioteca para manipulação e reconhecimento de imagens

# método classificador de carros

classificador = cv2.CascadeClassifier('cascades\\cars.xml')

# armazena imagem e transforma em preto e branco

imagem = cv2.imread('outros\\carro3.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# variavel com o método detecMultiScale que permite parametros para melhorar a probabilidade de acerto

detectado = classificador.detectMultiScale(imagemCinza, scaleFactor=1.01, minNeighbors=11)

# loop que coloriza pixels em vermelho em volta da face na forma de retangulo

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a),(0,0,255), 2)

    # exemplo de aplicação
    # condição que detecta veicúlos maiores que um determinado padrao estipulado para uma rodovia

    if l > 50 or a > 50:
        print("Veiculo fora do padrao")

cv2.imshow("Encontrado: ", imagem)
cv2.waitKey()
