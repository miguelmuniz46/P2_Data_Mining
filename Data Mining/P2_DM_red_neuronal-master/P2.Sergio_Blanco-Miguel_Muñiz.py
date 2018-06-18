import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import tensorflow as tf
np.random.seed()
 
class xorMLP(object):
    def __init__(self, learning_rate=0.):
        #En el constructor de la clase guardamos la constante aprendizaje que se nos ha proporcionado
        #Asi como también el valor de las entradas, que será el array de las X, con sus correspondientes salidas esperadas que serán las Y
        #Por otro lado asignamos las bias de las neuronas con valor 1 o -1, en este caso hemos escogido el valor -1
        self.aprendizaje=learning_rate
        self.X = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.Y = np.array([[0],[1],[1],[0]])
        self.bias=-1

        # Asignamos unos valores aleatorios (entre -1 y 1) para los pesos de las entradas que van a la neurona de la capa oculta
        rand = np.random.uniform(-1, 1, size=9)
        rand = rand.round(5)
        #capaOculta
        self.hide_w = np.array([(rand[0], rand[1],rand[2]), (rand[3],rand[4],rand[5])])

        # Asignamos unos valores aleatorios (entre -1 y 1) para los pesos de las entradas que van a la neurona de la capa de salida
        self.output_w = np.array([(rand[6],rand[7],rand[8])])


    def fit(self):

        #Establecemos un valor para el error pequeño, que será la condición de parada de entrenamiento de la red
        #En el caso de que la suma (en valor absoluto) de los errores para los 4 casos es más pequeño que el error que hemos establecido
        #quiere decir que la red de neuronas esta entrenada con cierta precisión
        errorMAX=0.1
        sumaEO = errorMAX
        k=0 #Variable que determinará cuantas iteracciones se han requerido para obtener el resultado
        done=False
       
        while not done:
              if sumaEO < errorMAX:#Condición de parada del entrenamiento de la red
                done=True

              else:
                    #Comienza el proceso de Feedfordward
                    sumaEO = 0

                    #Con este bucle se calcula por un lado  el sumatorio de todas las entradas de la neurona de la capa oculta por sus correspondientes pesos
                    #para calcular la función activación de esta neurona, que en este caso será la sigmoidal
                    for x,y in zip(self.X,self.Y):
                        x_temp=np.array(np.concatenate(([1],x)))
                        fAct_h = np.array([])
                        for neurona in self.hide_w:
                            temp = np.array([])
                            for peso in range(1, len(neurona), 1):
                                temp = np.append(fAct_h, 1 / (1 + np.exp(- np.dot(neurona, x_temp))))
                            fAct_h = np.append(fAct_h,sum(temp))

                        fAct_o = np.array([])
                        fAct_h_temp=np.array(np.concatenate(([1],fAct_h)))
                        for neurona in self.output_w:
                            temp = np.array([])
                            for peso in range(1, len(neurona), 1):
                                temp = np.append(fAct_o, 1 / (1 + np.exp(- np.dot(neurona, fAct_h_temp))))
                            fAct_o = np.append(fAct_o,sum(temp))


                        #Calculamos el error que hemos obtenido con el valor de salida de la neurona de salida, y la comparamos con el valor esperado y
                        eO=np.array([])
                        for neurona in fAct_o:
                            eO= np.append(eO,(neurona*(1-neurona))*(y[0] - neurona))
                        sumaEO += sum(abs(eO))#Actualizamos el valor de la suma de los errores de salida

                        #A continuación comienza el proceso de BackPropagation
                        old_output_w=self.output_w#Guardamos el valor del peso de la salida de la neurona de la capa de salida, ya que posteriormente lo tendremos que utilizar para calcular el error de la capa oculta

                        #incWho
                        #Actualizamos los pesos que van de las neuronas de la capa oculta a la neurona de la capa de salida
                        for neurona in range(0, len(self.output_w), 1):
                            for peso in range(1, len(self.output_w[neurona]), 1):
                                self.output_w[neurona][peso] += self.aprendizaje * eO * fAct_h[neurona]

                        ###############Es necesario incWio  porque no hay conexion directa i-o????????


                        #Calculamos el error de la capa oculta con el valor del peso que habiamos guardado previamente antes de actualizarlo
                        eh = np.array([])
                        for neurona in range(0,len(fAct_h),1):
                            eh = np.append(eh, (fAct_h[neurona] * (1 - fAct_h[neurona])) * (eO * old_output_w[0][neurona+1]))

                        old_hide_w=self.hide_w

                        #incWih
                        #Con el error de la capa oculta, a continuación, actualizamos los pesos que van de las neurona de la capa de entrada a la neurona de la capa oculta, así como tambien peso del bias
                        temp=np.array(np.concatenate(([0],x)))
                        for neurona in range(0, len(self.hide_w), 1):
                            for peso in range(0, len(self.hide_w[neurona]), 1):
                                self.hide_w[neurona][peso] += self.aprendizaje * eh[neurona] * x[neurona]


                        k+=1

        print ("Iteraciones totales: ",k)

 
    def predict(self, x):
        #Una vez que hemos terminado de entrenar a la red de neuronas, para comprobar la precisión de la red,
        #en el predict, le pasamos como parámetro el array con las entradas y así obtener una salida estimada
        """
        x = [x1, x2]
        """
        #Proceso para obtener el valor de la red de neuronas, con los pesos que habiamos guardado en el fit
        fAct_h = np.array([])
        for neurona in self.hide_w:
            temp = np.array([])
            for peso in range(1, self.hide_w[neurona], 1):
                temp = np.append(fAct_h, 1 / (1 + np.exp(- np.dot(neurona, x))))
            fAct_h = np.append(sum(temp))


        fAct_o = np.array([])
        for neurona in self.output_w:
            temp = np.array([])
            for peso in range(1, self.output_w[neurona], 1):
                temp = np.append(fAct_o, 1 / (1 + np.exp(- np.dot(neurona, fAct_h))))
            fAct_o = np.append(sum(temp))

        return fAct_o



class DeepMLP(object):
    def __init__(self, layers_size, learning_rate=0.):
        """
        ejemplos layers_size 
            [100, 50, 10] 100 neuronas de entrada, 50 de capa oculta 1 y 10 de salida
            [100, 50, 20, 10] 100 neuronas de entrada, 50 de capa oculta 1, 50 de capa oculta 2 y 10 de salida
            [100, 50, 20, 50, 10] etc.
        """
        self.layers_size = layers_size
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, [None, self.layers_size[
            0]])  # Placeholder que almacenara una matriz[X,Y] que puede tener cualquier cantidad de valores en X, en Y tiene que tener tantos valores como se indiquen en la primera capa de layers_size
        self.y = tf.placeholder(tf.float32, [None, self.layers_size[
            -1]])  # Igual que el anterior, pero en Y almacena la cantidad de valores indicados en la ultima capa de layers_size
        # Un placeholder es una variable a la que no se le asignaran datos en otro momento
        self.w = []  # Lista que almacena todos los pesos de la red
        self.b = []  # Lista que almacena todos los Bias de la red
        for i in range(len(self.layers_size) - 1):
            self.w.append(tf.Variable(2 * np.random.random((self.layers_size[i], self.layers_size[i + 1])) - 1,
                                      dtype=tf.float32))  # inicializacion de los pesos con valores aleatorios de -1 a 1, se agrupan en matrices de tamaño nºnodosEntrada * nºnodosSalida
            self.b.append(tf.Variable(2 * np.random.random((1, self.layers_size[i + 1])) - 1,
                                      dtype=tf.float32))  # igual que los pesos pero en este caso solo se almacenan vectores de tamaño nºnodosSalida

        # Feedfordward de la red
        self.yHat = self.x
        for i in range(len(self.layers_size) - 1):  # Se realiza la operacion tantas veces como capas haya
            self.yHat = tf.matmul(self.yHat, self.w[i]) + self.b[i]  # salida de la capa = entrada*pesos+Bias
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                               logits=self.yHat))  # Determina la perdida que tiene el modelo, nos dice como de ineficiente son las predicciones de la red, ademas le aplica directamente la funcion de activacion softmax
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            cross_entropy)  # Minimiza el crozz_entropy usando un descenso de gradiente

        self.sess = tf.InteractiveSession()  # Crea una sesion de Tensorflow
        self.sess.run(tf.initialize_all_variables())  # inicializa todas la varibles creadas en tensorflow
        # en mi caso estoy usando tf.initialize_all_variables() y aque estoy con la version 0.8 de TensorFlow, en una version mas actual se puede cambiar por tf.global_variables_initializer()

    def fit(self, X):
        """
        X = entradas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        """
        porcentageAlcanzado = False  # bool que nos indica si se han alcanzado el % de acierto dado
        i = 0  # Contador de iteraciones para evitar bucles infinitos
        batchX, batchY = X.next_batch(
            100)  # Se cogen un batch con 100 elementos aleatorios dentro de X y se dividen en datos de entrada batchX y datos esperados de salida batchY
        self.score(batchX, batchY)  # Llama a la funcion score para inicializar las variables necesarias
        while porcentageAlcanzado == False:
            i += 1
            if i % 100 == 0:  # cada 100 iteraciones se hace una actualizacion de la precision del entrenamiento
                train_accuracy = self.accuracy.eval(
                    feed_dict={self.x: batchX, self.y: batchY})  # Se evalua actualiza el valor de accuracy
                print('step %d, training accuracy %g' % (i, train_accuracy))
                if train_accuracy >= 0.98:  # Si la precision alcanza un valor dado paramos el entrenamiento
                    porcentageAlcanzado = True
            self.sess.run(self.train_step,
                          feed_dict={self.x: batchX, self.y: batchY})  # Se actualiza el valor de train_step
            if i >= 100000:  # En el caso de llegar a 100000 iteraciones se para el entrenamiento
                porcentageAlcanzado = True
            batchX, batchY = X.next_batch(100)  # Se coge el siguiente batch para usar en la siguiente iteracion
        print("Iteraciones: %d" % i)


    def score(self, X, Y):
        """
        X = entradas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        """
        correct_prediction = tf.equal(tf.argmax(self.yHat, 1), tf.argmax(self.y,
                                                                         1))  # Se determina si la predcicion realizada en yHat es correcta. yHat e Y son dos arrays de 10 elementos, cada elemento dentro del array determinala probabilidad de que el numero a predecir sea el indice del array es decir, [0,0,1] el resultado seria 2. argmax coge el valor mas grande dentro del array, por lo que se comparan el valor de yHat con mas probabilidad frente al valor real de y
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                               tf.float32))  # la prediccion se saca cogiendo el array de booleanos anterior, se castea a float para poder hacer la media, ek resultado nos determina cual es la probabilidad de acertar
        self.a = self.sess.run(self.accuracy, feed_dict={self.x: X,
                                                         self.y: Y})  # Se ejecuta accuracy dentro de Tensorflow con los valores de testeo X e Y
        print('accuracy %g' % self.a)


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # TODO MNIST TESTS
    print("DeepMLP:")
    dpml = DeepMLP([784, 500, 500, 2000, 30], 0.2)
    X_train = mnist.train
    dpml.fit(X_train)
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    dpml.score(X_test, Y_test)



