# LAB3.0

NUM_CLASSES = 6

____________________________________________________________________________________
  ## Прогон (0). Cлучайно инициализированная нейронная сеть MobileNetV2

        base_model = tf.keras.applications.MobileNetV2(
                                  include_top=False,
                                  weights=None, classe = 6)
      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/1800457221553b4080c52716d1af84a2b3590b2a0/tensorboard/epoch_categorical_accuracy(0).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/1800457221553b4080c52716d1af84a2b3590b2a0/tensorboard/epoch_loss(0).svg)
   
  
  ## Прогон (1). Замена классификатора в предобученной сети на необходимый для решения поставленной задачи. 

  lr = 0.00001
  
        base_model = tf.keras.applications.MobileNetV2(
                                  include_top=False,
                                  weights='imagenet')
        global_average_layer =  tf.keras.layers.GlobalAveragePooling2D()     
        prediction_layer =    tf.keras.layers.Dense(6, activation=tf.keras.activations.softmax)

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/7a555d6e36d48af6b30b831a624fabf8d13ce41c/tensorboard/epoch_categorical_accuracy%20(1).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/7a555d6e36d48af6b30b831a624fabf8d13ce41c/tensorboard/epoch_loss%20(1).svg)
   
   
  ## Прогон (2). Сеть (1) с измененным lr 
  
  lr = 0.000001
       
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/7a555d6e36d48af6b30b831a624fabf8d13ce41c/tensorboard/epoch_categorical_accuracy%20(2).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/7a555d6e36d48af6b30b831a624fabf8d13ce41c/tensorboard/epoch_loss%20(2).svg)
        
  ## Прогон (3). Сеть (1). Веса MobileNetV2 заморожены. Сеть сохраняется для пункта (4)
  
        base_model.trainable = False
  
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_categorical_accuracy%20(3).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_loss%20(3).svg)
        
  ## Прогон (4). Загруженная сеть (3). Веса MobileNetV2 разморожены для дообучения

         base_model.trainable = True
      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_categorical_accuracy%20(4).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_loss%20(4).svg)
    
   ## Прогон (5). Веса MobileNetV2 заморожены. Сеть сохраняется для пункта (6)

         lr = 0.00004
         base_model.trainable = False
      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_categorical_accuracy%20(5).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_loss%20(5).svg)
  
  
  ## Прогон (6). Загруженная сеть (5). Веса MobileNetV2 разморожены для дообучения

         base_model.trainable = True
      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_categorical_accuracy%20(6).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab3/1a9f443f1fff71f85f4d49e4d821eddd707e52b4/tensorboard/epoch_loss%20(6).svg)
  
  
