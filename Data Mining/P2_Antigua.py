import numpy as np
import os

# import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf


class xorMLP(object):
    def __init__(self, learning_rate=0.):
        self.aprendizaje = learning_rate
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.Y = np.array([[0], [1], [1], [0]])
        self.bias = -1

    def fit(self):
        ###################################################### Capa oculta 1 ###########################################################################
        rand = round(np.random.uniform(-1, 1), 5)
        rand2 = round(np.random.uniform(-1, 1), 5)
        capaOculta1 = np.array([rand, rand2])

        rand = round(np.random.uniform(-1, 1), 5)
        pesoBias_1_Oculta = np.array([rand])
        biasCapaOculta1 = self.bias

        #################################################### Capa oculta 2 ######################################################################
        rand = round(np.random.uniform(-1, 1), 5)
        rand2 = round(np.random.uniform(-1, 1), 5)
        capaOculta2 = np.array([rand, rand2])

        rand = round(np.random.uniform(-1, 1), 5)
        pesoBias_2_Oculta = np.array([rand])
        biasCapaOculta2 = self.bias

        ################################################### Capa salida #####################################################################
        rand = round(np.random.uniform(-1, 1), 5)
        rand2 = round(np.random.uniform(-1, 1), 5)
        capaSalida = np.array([rand, rand2])

        rand = round(np.random.uniform(-1, 1), 5)
        pesoBiasSalida = np.array([rand])
        biasCapaSalida = self.bias

        errorMAX = 0.10
        sumaEO = errorMAX
        k = 0
        done = False

        while not done:
            if sumaEO < errorMAX or k == 1000000:
                done = True

                self.pesosH1 = np.array(capaOculta1)
                self.pesoBiasH1 = np.array(pesoBias_1_Oculta)

                self.pesosH2 = np.array(capaOculta2)
                self.pesoBiasH2 = np.array(pesoBias_2_Oculta)

                self.pesosO = np.array(capaSalida)
                self.pesoBiasO = np.array(pesoBiasSalida)

            else:
                sumaEO = 0

                for x, y in zip(self.X, self.Y):

                    ###################################### Funcion activacion para oculta 1 #############################################
                    suma = biasCapaOculta1 * pesoBias_1_Oculta
                    for i in range(0, 2, 1):
                        suma += ((x[i] * capaOculta1[i]))

                    funcionActivacionOculta1 = 1 / (1 + np.exp(-suma))

                    ###################################### Funcion activacion para oculta 2 #############################################
                    suma = biasCapaOculta2 * pesoBias_2_Oculta
                    for i in range(0, 2, 1):
                        suma += ((x[i] * capaOculta2[i]))

                    funcionActivacionOculta2 = 1 / (1 + np.exp(-suma))

                    ###################################### Funcion activacion para salida #############################################
                    suma = biasCapaSalida * pesoBiasSalida
                    suma += capaSalida[0] * funcionActivacionOculta1
                    suma += capaSalida[1] * funcionActivacionOculta2
                    funcionActivacionSalida = 1 / (1 + np.exp(-suma))

                    ###################################### Calculamos el error de salida ##############################################
                    errorSalida = (funcionActivacionSalida * (1 - funcionActivacionSalida)) * (
                                y[0] - funcionActivacionSalida)

                    sumaEO += abs(errorSalida)

                    valorEntradaHO_1 = capaSalida[0]
                    valorEntradaHO_2 = capaSalida[1]

                    ##################################################incWho de la neurona oculta 1 ######################################################
                    capaSalida[0] += self.aprendizaje * errorSalida * funcionActivacionOculta1

                    ##################################################incWho de la neurona oculta 2 ######################################################
                    capaSalida[1] += self.aprendizaje * errorSalida * funcionActivacionOculta2

                    ################################################### error de la neurona oculta 1 #######################################################
                    errorH_1 = (funcionActivacionOculta1 * (
                                1 - funcionActivacionOculta1)) * errorSalida * valorEntradaHO_1

                    ################################################### error de la neurona oculta 2 #######################################################
                    errorH_2 = (funcionActivacionOculta2 * (
                                1 - funcionActivacionOculta2)) * errorSalida * valorEntradaHO_2

                    ################################################### incWih de la neurona oculta 1 ########################################################
                    capaOculta1[0] += self.aprendizaje * errorH_1 * x[0]
                    capaOculta1[1] += self.aprendizaje * errorH_1 * x[1]
                    pesoBias_1_Oculta += self.aprendizaje * errorH_1 * biasCapaOculta1

                    ################################################### incWih de la neurona oculta 2 ########################################################
                    capaOculta2[0] += self.aprendizaje * errorH_2 * x[0]
                    capaOculta2[1] += self.aprendizaje * errorH_2 * x[1]
                    pesoBias_2_Oculta += self.aprendizaje * errorH_2 * biasCapaOculta2

                    k += 1

        print("Iteraciones totales: ", k)

    def predict(self, x):
        # Una vez que hemos terminado de entrenar a la red de neuronas, para comprobar la precisión de la red,
        # en el predict, le pasamos como parámetro el array con las entradas y así obtener una salida estimada
        """
        x = [x1, x2]
        """
        # Proceso para obtener el valor de la red de neuronas, con los pesos que habiamos guardado en el fit
        suma = self.pesoBiasH1 * self.bias
        for i in range(0, 2, 1):
            suma += ((x[i] * self.pesosH1[i]))
        fActOculta1 = 1 / (1 + np.exp(-suma))

        suma = self.pesoBiasH2 * self.bias
        for i in range(0, 2, 1):
            suma += ((x[i] * self.pesosH2[i]))
        fActOculta2 = 1 / (1 + np.exp(-suma))

        suma = self.pesoBiasO * self.bias
        suma += self.pesosO[0] * fActOculta1
        suma += self.pesosO[1] * fActOculta2
        fActSalida = 1 / (1 + np.exp(-suma))

        # Devolvemos el valor de salida estimado, redondeado hasta 3 decimales
        # return round(fActSalida, 3)
        return fActSalida


class DeepMLP(object):
    def __init__(self, layers_size, learning_rate=0.):
        """
        ejemplos layers_size
            [100, 50, 10] 100 neuronas de entrada, 50 de capa oculta 1 y 10 de salida
            [100, 50, 20, 10] 100 neuronas de entrada, 50 de capa oculta 1, 50 de capa oculta 2 y 10 de salida
            [100, 50, 20, 50, 10] etc.
        """
        pass

    def fit(self, X, Y):
        """
        X = entradas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        """
        pass

    def score(self, X, Y):
        """
        X = entradas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        """
        pass


if __name__ == '__main__':
    #from tensorflow.examples.tutorials.mnist import input_data

    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # TODO MNIST TESTS

# Pruebas para xorMLP
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    red = xorMLP(0.2)
    red.fit()

    for i in range(0, 4, 1):
        x = np.array(X[i])
        print("X:", x)
        print("Y:", Y[i])
        res = red.predict(x)
        print("Y de la red: ", res)
        print("Y esperada: " , round(res[0], 0))
