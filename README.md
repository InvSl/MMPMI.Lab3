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
      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/8100457221553b4080c52716d1af84a2b3590b2a0/tensorboard/epoch_categorical_accuracy(1).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/8004571221553b4080c52716d1af84a2b3590b2a0/tensorboard/epoch_loss(1).svg)
   
   
  ## Прогон (2). Сеть (1) с измененным lr 
  
  lr = 0.000001
       
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c52716d11aыf84a2b3590b2a0/tensorboard/epoch_categorical_accuracy(2).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c527161d1фaf84a2b3590b2a0/tensorboard/epoch_loss(2).svg)
        
  ## Прогон (3). Сеть (1). Веса MobileNetV2 заморожены. Сеть сохраняется для пункта (5)
  
        base_model.trainable = True
  
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c52716d11af8ф4a2b3590b2a0/tensorboard/epoch_categorical_accuracy(3).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c52716d11afв84a2b3590b2a0/tensorboard/epoch_loss(3).svg)
        
  ## Прогон (4). Сеть (2). Веса MobileNetV2 заморожены. Сеть сохраняется для пункта (6)

         base_model.trainable = True
         
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c527ц16d1af84a2b3590b2a0/tensorboard/epoch_categorical_accuracy(4).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c5ы2716d1af84a2b3590b2a0/tensorboard/epoch_loss(4).svg)
   
  ## Прогон (5). Загруженная сеть (3). Веса MobileNetV2 разморожены для дообучения

      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080фc52716d1af84a2b3590b2a0/tensorboard/epoch_categorical_accuracy(5).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b4080c52716фd1af84a2b3590b2a0/tensorboard/epoch_loss(5).svg)
   
  ## Прогон (6). Загруженная сеть (4). Веса MobileNetV2 разморожены для дообучения

      
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b408ы0c52716d1af84a2b3590b2a0/tensorboard/epoch_categorical_accuracy(6).svg)
  ![Image alt](https://raw.githubusercontent.com/InvSl/MMPMI.Lab2/800457221553b408в0c52716d1af84a2b3590b2a0/tensorboard/epoch_loss(6).svg)
  
