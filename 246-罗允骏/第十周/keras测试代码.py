from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


def ImageToMatrix(filename):
    im = Image.open(filename)
    #change to greyimage
    im=im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='int')
    return data

# model = load_model('/Users/zhang001/Desktop/my_model2.h5')
model = load_model('path_to_my_model.h5')
# model.save('/Users/zhang001/Desktop/my_model2.h5')


while 1:
    i = input('number:')
    j = input('type:')
    data = ImageToMatrix('/Users/zhang001/Desktop/picture/'+str(i)+'_'+str(j)+'.png')
    data = np.array(data)
    data = data.reshape(1, 28, 28, 1)
    print ('test['+str(i)+'_'+str(j)+'], num='+str(i)+':')
    # print model.predict_classes(
    #     data, batch_size = 1 , verbose = 0
    # )