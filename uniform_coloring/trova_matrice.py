import os # import per prelevare i le immagini da analizzare
import cv2 # import per leggere le immagini
import numpy as np #per la lavorazione delle immagini
from emnist import list_datasets # importato per caricare il dataset di emnist letters
import matplotlib.pyplot as plt # usato per il disegno delle immagini con la grafica
import shelve # usata per la scrittura/lettura di una matrice in un file
# Download images of letters from training samples or test samples
from emnist import extract_test_samples # importo la funzione per scaricare il dataset test
from emnist import extract_training_samples # importo la funzione per scaricare il dataset test
import tensorflow as tf # imorto tensorflow per creare il modello
from pathlib import Path
import operator
from collections import OrderedDict

input_image_path = None
input_dimension = None
lista_numeri=[]

#estraggo dal dataset solamente le lettere che mi servono (B,G,Y,T)
def estrai_lettere(images, labels, letters):
    """
    INPUT:
    letters --> sono le lettere che voglio estrarre dal dataset
    labels --> sono le etichette associate ad ogni immagine
    images --> sono le immagini che devono essere elaborate
    OUTPUT:
    etichette --> contenente le nuove etichette associate ad ogni immagine
    array --> numpy.ndarray contenente le immagini da elaborare
    NOTE AGGIUNGIVE:
    le etichette restituite sono riscritte come 0,1,2,3 dove 0 = B, 1 = G, 2 = T, 3 = Y //ordine alfabetico
    """

    etichette=[]#etichette delle varie lettere B,T,G,Y
    posizione_lettere=[]
    lettere={2:0,7:1,20:2,25:3}#lettere B,T,G,Y
    lettere_trovate={0:0,1:0,2:0,3:0}
    count=0
    for indice in labels:
        if(indice in letters):
            etichette.append(lettere[indice])

            lettere_trovate[lettere[indice]]+=1
            posizione_lettere.append(count)
        count+=1

    array=np.empty((0,images.shape[1],images.shape[2]),dtype=np.uint8)
    print(lettere_trovate)
    count=0
    for indice in posizione_lettere:
        """
        print(f"{count}:{indice}")
        print(f"lettera: {etichette[count]}")
        cv2.imshow("",images[indice])
        cv2.waitKey()
        """
        """
        print(f"\n\n\n\ndim1: {images[indice].shape} dim2: {type(images[indice][0][0])} ")
        """
        array=np.append(array,np.array([images[indice]]),axis=0)
        """
        for value in array:
            print(f"\n\n\n\ndim1: {value.shape} dim2: {type(value[0][0])} ")
            print(images[indice].shape)
            print(images[indice])
            cv2.imshow("",images[indice])
            cv2.waitKey()
            print(value.shape)
            print(value)
            cv2.imshow("",value)
            cv2.waitKey()
        """
        #print(count,indice,array.shape)
        #print(f"array: {array[count]}")
        count+=1
    array = np.asarray(array)
    etichette = np.asarray(etichette)

    return array, etichette

""" da fare funzione ordinamento"""
def ordinamento(lista_lettere,posizione_lettere):

    array_iniziale=[(elemento[0],elemento[1],lista_lettere[posizione_lettere.index(elemento)]) for elemento in posizione_lettere]
    """for x,y,immagine in array_iniziale:
        print(f"coordinate ({x};{y})")
        plt.imshow(immagine),plt.show()
    """
    xi,yi,imgi=array_iniziale[0]
    righeTrovate={yi:[(xi,yi,imgi)]}

    for x,y,img in array_iniziale[1:]:
        successivo=0
        mean_list=list(righeTrovate.keys())
        while True:
            diff=y-mean_list[successivo]
            print(f"y-meanlist[{successivo}] vale: \n{y}-{mean_list[successivo]} = {diff}")
            if(abs(diff)<15):
                # allora la lettera si trova nella stessa riga del valor medio in questione
                #devo cambiare l'indice e aggiungere l'elemento
                new_list=[]
                new_list=righeTrovate.pop(mean_list[successivo])
                new_list.append((x,y,img))
                #ordino la lista
                new_list.sort(key=lambda x: x[0])
                tmp=[y for x,y,img in new_list]
                y_mean=sum(tmp)/len(tmp)
                righeTrovate[y_mean]=new_list


                break
            #allora non è nella stessa riga perciò vado a controllare nelle altre
            successivo+=1
            print(f"successivo= {successivo}, righeTrovate lenght {len(righeTrovate)} , mean length {len(mean_list)}" )
            if(successivo >= len(righeTrovate)):
                #allora non ci sono più righe da leggere quindi devo aggiungere la y come nuova riga
                righeTrovate[y]=[(x,y,img)]
                break
    """
    for value in righeTrovate.values():

        for x,y,img in value:
            print(x,y)

            plt.imshow(img),plt.show()
    """
    dict_definitivo={}
    for value in sorted(righeTrovate.keys()):
        dict_definitivo[value]=righeTrovate[value]

    return [img for value in dict_definitivo.values() for x,y,img in value ]

def preleva_input():
    global input_dimension
    print("inserisci righe matrice")
    righe=input()
    print("inserisci colonne matrice")
    colonne=input()
    input_dimension=(int(righe),int(colonne))
    print(f"la matrice ha dimensioni {input_dimension}")

    while True:
        print(f"inserisci percorso immagine o inserisci il file nella cartella:\n\
        {os.path.abspath(os.getcwd())}/immagini/input")
        print("se vuoi usare il default invia Y")
        input_image_path=input()
        if(input_image_path=='y'or input_image_path=='Y'):
            file=""
            input_image_path=f"{os.path.abspath(os.getcwd())}/immagini/input/"
            print("inserisci nome del file, anche il formato")
            print(os.listdir("./immagini/input/"))
            file=input()
            input_image_path+=file

        print(input_image_path)
            #cv2:setBreakOnError(true);#usato per evitare che imread esca dal programma, adesso imread passa all' Exception
        img=cv2.imread(input_image_path)

        if(type(img).__name__ != 'ndarray'):
            print("Errore caricamento dell'immagine non riuscito, riprovare\n\
            percorso inserito ",input_image_path)
        else:
            print("immagine aperta con successo")
            break
    return img,input_dimension

def ritaglia_immagine(immagine):
    global input_dimension
    img=immagine.copy()
    height,width,c=img.shape
    linee_orizzontali=np.zeros((height,width),dtype='uint8') ##ci inserisco i punti bianchi dell'immagine
    linee_verticali=(width,0)##ci inserisco i punti bianchi dell'immagine
    print(height,width,c)
    img = cv2.Canny(img,255,255/3)
    #print(f"ci sono arrivato {img}")
    array=np.array(img)

    indici=[]
    conta_passi=0
    for cella in img:
        if(any(cella)):
            conta_bianchi=0
            conta_scorrimento=0
            for colori in cella:
                if(colori==255):
                    if(conta_scorrimento<linee_verticali[0]):#corrisponde all'inizio
                        linee_verticali=(conta_scorrimento,linee_verticali[1])
                    if(conta_scorrimento>linee_verticali[1]):#corrisponde alla fine
                        linee_verticali=(linee_verticali[0],conta_scorrimento)
                    #print(f"conta= {conta_bianchi} numero={colori}")
                    conta_bianchi+=1
                conta_scorrimento+=1
            if(conta_bianchi > -1):
                indici.append(conta_passi)
            #print(f"{cella}:{conta_passi}")
        #se non è presente nessun valore != 0 allora è una linea bianca
        conta_passi+=1
    primo=indici[0]-2
    ultimo=indici[len(indici)-1]+2

    value_precedente=0
    count=0
    ritaglio_immagine=np.full((ultimo-primo,(linee_verticali[1]-linee_verticali[0])+4,3),255,dtype='uint8')
    linee_orizzontali=np.zeros((ultimo-primo,(linee_verticali[1]-linee_verticali[0])+4),dtype='uint8')
    lettere=np.zeros((ultimo-primo,(linee_verticali[1]-linee_verticali[0])+4),dtype='uint8')

    for value in indici:
        ritaglio_immagine[value-primo]=immagine[value][linee_verticali[0]-2:linee_verticali[1]+2][:]


    #print(f"{50*input_dimension[0]},{50*input_dimension[1]}")

    ritaglio_immagine=cv2.resize(ritaglio_immagine,(65*input_dimension[1],65*input_dimension[0]),interpolation=cv2.INTER_NEAREST )

    return ritaglio_immagine

def trova_lettere(im):
    global input_dimension

    im3 = im.copy()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    posizione_lettere=[]
    #troviamo i contorni delle lettere
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    lista_lettere=[]
    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>15 and h <50:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y-2:y+h+2,x-2:x+w+2]
                #controllo per verificare che una lettera non venga contata due volte
                posizione_lettere.append(((x,y),(x+w,y+h)))
                        #allora l'immagine è doppione
                #plt.imshow(roi)
                #plt.show()
                roismall = cv2.resize(roi,(28,28))
                lista_lettere.append(roismall)
                plt.imshow(im)
                plt.show()



    if(len(posizione_lettere)>input_dimension[0]*input_dimension[1]):
        #allora ci sono immagini in più, vanno eliminate
        count=0
        for element in posizione_lettere[:len(posizione_lettere)-1]:
            secondo_elemento=0
            for i in posizione_lettere[count+1:]:
                #allora ancora c'è un elmento successivo
                if(element[0][0]>i[0][0] and element[0][1]>i[0][1] and element[1][0]<i[1][0] and element[1][1]<i[1][1]):
                    print("l'elemento doppione trovato ",element,i)
                    print(f"{element[0]}>{i[0]},{element[1]}<{i[1]}")
                    plt.subplot(121),plt.imshow(lista_lettere[count])
                    plt.subplot(122),plt.imshow(lista_lettere[posizione_lettere.index(i)])
                    plt.show()
                    del posizione_lettere[count]
                    del lista_lettere[count]

                    count-=1
                secondo_elemento+=1
            count+=1

    if(len(posizione_lettere)<input_dimension[0]*input_dimension[1]):
        print("ERRORE: alcune lettere non sono state trovate")
        plt.imshow(im)
        plt.show()
        return "errore" , "errore"

    responses = np.array(responses,np.uint8)
    responses = responses.reshape((responses.size,1))
    count=0

    lista_lettere=ordinamento(lista_lettere,[valueXY for valueXY,valueXYWH in posizione_lettere])


    print(f"salvataggio delle immagini nella cartella {os.path.abspath(os.getcwd())}/immagini/output/\ndimensione immagini 28x28")
    for img in lista_lettere:
        cv2.imwrite(f'immagini/output/immagine{count}.jpeg',img)
        #plt.imshow(img)
        #plt.show()
        count+=1
    """
    img_number=0
    array_immagini=[]
    while os.path.isfile(f"./immagini/output/immagine{img_number}.jpeg"):
        print(posizione_lettere[img_number])
        img=cv2.imread(f"./immagini/output/immagine{img_number}.jpeg")
        array_immagini.append(img)
        img_number+=1
    """
    return im3,posizione_lettere

def mean_pred(y_true,y_pred):
    return tf.keras.backend.mean(y_pred)

def lettura_rete_neurale():
    global lista_numeri
    images_training, labels_training = extract_training_samples('letters')
    images_test,labels_test= extract_test_samples('letters')
    model = tf.keras.Sequential()
    #aggiungo i livelli della rete neurale
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(784,activation='relu'))
    model.add(tf.keras.layers.Dense(98,activation='relu'))
    model.add(tf.keras.layers.Dense(4,activation='softplus'))
    model.summary()
    file= shelve.open('matrice.dat')   #crea il file matrice.dat che conterrà il dizionario

    lista_nomi_file=['images_training','labels_training','images_test','labels_test']
    count=0
    ##lettura dei dati da file se gia è stata fatta l'operazione di filtraggio dei dati senno le ricerco e le salvo nel file
    if(file.keys()!=[]):#KeysView
        images_training=file[lista_nomi_file[0]]
        labels_training=file[lista_nomi_file[1]]
        images_test=file[lista_nomi_file[2]]
        labels_test=file[lista_nomi_file[3]]
        file.close()
    else:
        #allora devo caricare le lettere
        #lettere B,G,T,Y
        lettere=[2,7,20,25]

        print("estrazione delle lettere T,B,G,Y del training")
        images_training , labels_training=estrai_lettere(images_training,labels_training,lettere)
        print("estrazione delle lettere T,B,G,Y del training")


        print("estrazione delle lettere T,B,G,Y del test")
        images_test, labels_test=estrai_lettere(images_test, labels_test,lettere)#le x sono images e le y sono le labels
        print("estrazione delle lettere T,B,G,Y del test")
        #assegno le matrici al file
        file['images_training']=images_training
        file['labels_training']=labels_training
        file['images_test']=images_test
        file['labels_test']=labels_test
        file.close()  #chiudo il file dizionario
    #trasformo le immagini in RGB in immagini con una sola scala di colori
    images_training=images_training.reshape(images_training.shape[0],28,28)
    images_test=images_test.reshape(images_test.shape[0],28,28)

    #normalizzo i pixel
    images_training=images_training/255.0
    images_test=images_test/255.0

    #stampa e setta la loss fuction e la metrica da utilizzare, optimizer non capito
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    #alleno il modello con i dati del dataset
    model.fit(images_training,labels_training, epochs=2)

    #salvo il modello
    model.save('letturaScritte.model')
    #carico il modello
    model=tf.keras.models.load_model('letturaScritte.model')

    #faccio le operazioni di test per vedere se il modello ha fatto overfitting
    loss,accuracy=model.evaluate(images_test,labels_test)

    #ora che il modello è stato creato possiamo proseguire con la lettura delle immagini che bisogna leggere
    img_number=0
    while os.path.isfile(f"./immagini/output/immagine{img_number}.jpeg"):
      try:
        #lettura immagine
        img=cv2.imread(f"./immagini/output/immagine{img_number}.jpeg")[:,:,0]
        #resize dell'immagine
        img=cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA )
        #aggiunta del singolo canale di colore
        img=img.reshape(28,28)
        img=np.array([img])


        #normalizzazione dell'immagine
        img_normalize=np.array(img/255.0)

        #print(img_normalize.shape)
        #lista del valore secondo il modello creato,lista di valori
        prediction=model.predict(img_normalize)
        print(prediction)

        #aggiungo l'elemento alla lista dei numeri trovati
        lista_numeri.append(np.argmax(prediction))
        #visualizzo il numero trovato
        dict={2:'T',0:'B',1:'G',3:'Y'}
        print(f"the digit is probably a {dict[np.argmax(prediction)]}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
      except:
        print("Error!!")
      finally:
        img_number+=1

    return lista_numeri

"""
def main():
    global lista_numeri
    global mat_input
    lista_colori=[]
    immagine=preleva_input()
    plt.imshow(immagine)
    plt.show()

    immagine=ritaglia_immagine(immagine)
    plt.imshow(immagine)
    plt.show()

    immagine, lettere_trovate=trova_lettere(immagine)
    if(type(immagine).__name__!="ndarray"):
        exit()

    lettura_rete_neurale()

    print(lista_numeri)
    for elemento in lista_numeri:
        if elemento==0:
            lista_colori.append('B')
        elif elemento==1:
            lista_colori.append('G')
        elif elemento==2:
            lista_colori.append('T')
        elif elemento==3:
            lista_colori.append('Y')
    print(lista_colori)

    mat_input=np.zeros((input_dimension[0],input_dimension[1]))
    mat_input=np.array(mat_input,dtype=str)
    contatore_lista=0
    x=0
    y=0
    while x<input_dimension[0]:
        y=0
        while y<input_dimension[1]:
            mat_input[x,y]=lista_colori[contatore_lista]
            contatore_lista+=1
            y+=1
        x+=1
    print("mat_input: ",mat_input)
    img_number=0
    while os.path.isfile(f"./immagini/output/immagine{img_number}.jpeg"):
        file_path = Path(f'./immagini/output/immagine{img_number}.jpeg')
        try:
            file_path.unlink()
        except OSError as e:
            print("Error: %s : %s" % (file_path, e.strerror))
        img_number+=1
if __name__=="__main__":
    main()
"""
