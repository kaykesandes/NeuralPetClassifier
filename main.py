import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import urllib.request
import zipfile
import random 
import keras

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*", category=UserWarning)


dataset_url = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
zip_filename = "kagglecatsanddogs_5340.zip"
zip_path = os.path.join(os.getcwd(), zip_filename)

# Baixa o ZIP se ainda não existir
if not os.path.exists(zip_path):
    print("Baixando dataset...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    print("Download concluído:", zip_path)

# Extrai o conteúdo se a pasta PetImages não existir
extract_dir = os.path.join(os.getcwd(), "PetImages")
if not os.path.isdir(extract_dir):
    print("Extraindo dataset...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(os.getcwd())
    print("Extração concluída em:", extract_dir)
    # Remove o arquivo zip após a extração
    try:
        os.remove(zip_path)
        print(f"Arquivo zip removido: {zip_path}")
    except Exception as e:
        print(f"Erro ao remover o zip: {e}")

#Função que carrega img e retora ela mesmo e o um vetor de entrada
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


#Carregar todas img na pastas raiz
categories = ["Cat", "Dog"]
limite_por_categoria = 1000
data = []
for c, petimagem in enumerate(categories):
    pasta = os.path.join("PetImages", petimagem)
    print(f"Verificando pasta: {pasta}")  # Adicionado para debug
    imagens = [os.path.join(dp, f) for dp, dn, filenames 
               in os.walk(pasta) for f in filenames
               if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
    imagens = imagens[:limite_por_categoria]
    print(f"Imagens encontradas em {pasta}: {len(imagens)}")  # Adicionado para debug
    for img_path in imagens:
        try:
            img, x = get_image(img_path)
            data.append({'x': np.array(x[0]), 'y': c})
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")
            try:
                os.remove(img_path)
                print(f"Arquivo removido: {img_path}")
            except Exception as rem_err:
                print(f"Erro ao remover {img_path}: {rem_err}")
            
num_classes = len(categories)

random.shuffle(data)

# # Exibe o total de imagens carregadas
# print(f"Total de imagens carregadas: {len(data)}")

# # Exibe a quantidade por categoria
# cats = sum(1 for d in data if d['y'] == 0)
# dogs = sum(1 for d in data if d['y'] == 1)
# print(f"Imagens de gatos: {cats}")
# print(f"Imagens de cachorros: {dogs}")

# # Exibe um exemplo de cada categoria
# print("Exemplo de entrada (gato):", data[0]['x'].shape, "Classe:", data[0]['y'])
# print("Exemplo de entrada (cachorro):", data[-1]['x'].shape, "Classe:", data[-1]['y'])



train_split, val_split = 0.7, 0.15

#Criando Treinamento
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

#separa dados por rotulos
x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
print(y_test)


#Pre-processa os dados como antes, certificando que sejam float32 e normalizados entre 0 e 1.

# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

#oq temos
# print("finished loading %d images from %d categories"%(len(data), num_classes))
# print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
# print("training data shape: ", x_train.shape)
# print("training labels shape: ", y_train.shape)


#ver alguams imgs q temos
root_cat = "PetImages/Cat"
root_dog = "PetImages/Dog"

images_cat = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_cat) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg']]
images_dog = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dog) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg']]

images = images_cat + images_dog  # Junta as duas listas

idx = [int(len(images) * random.random()) for i in range(15)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(concat_image)
plt.show()


img_size = (224, 224)
batch_size = 32
num_classes = 2
epochs = 10
data_dir = "PetImages"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=img_size + (3,)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_val, y_val)
)

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

inp = vgg.input

new_classification_layer = Dense(num_classes, activation='softmax')

out = new_classification_layer(vgg.layers[-2].output)

model_new = Model(inp, out)

for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model_new.summary()

history2 = model_new.fit(x_train, y_train, 
                         batch_size=128, 
                         epochs=10, 
                         validation_data=(x_val, y_val))


fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.plot(history2.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.plot(history2.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

num_exemplos = 5
total = len(x_test)
atual = 0

num_exemplos = 5
total = len(x_test)
atual = 0

while atual < total:
    plt.figure(figsize=(15, 6))
    indices = range(atual, min(atual + num_exemplos, total))
    for i, idx in enumerate(indices):
        img = x_test[idx]
        true_label = np.argmax(y_test[idx])
        pred_label = np.argmax(model_new.predict(img[np.newaxis, ...]))
        classe_pred = "Cat" if pred_label == 0 else "Dog"
        classe_true = "Cat" if true_label == 0 else "Dog"
        plt.subplot(1, num_exemplos, i+1)
        plt.imshow(image.array_to_img(img * 255))
        plt.axis('off')
        plt.title(f"Pred: {classe_pred}\nTrue: {classe_true}", fontsize=14)
    plt.tight_layout()
    plt.show()
    atual += num_exemplos
    if atual < total:
        resposta = input("Pressione Enter para ver mais 5 exemplos ou digite 'sair' para encerrar: ")
        if resposta.strip().lower() == "sair":
            break