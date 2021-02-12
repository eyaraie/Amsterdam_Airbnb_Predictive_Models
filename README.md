# Amsterdam_Airbnb_predictive_models
An accommodation price depends on an ocean of factors. we can characterize them in three most important categories: 1) accommodation-based features, 2) neighborhood-based features features and 3) owner-based features.
Accommodation-based features can be called intrinsic features of an accommodation, some things like the number of bedrooms, its ameninty, etc. Neighborhood-based features are properties like the distance from city center, caffe, touristic places, its walk-ability and in general every external factor which can influence the price of an accommodation. Owner-based features can be named to features which come from interaction of owner and applicant for instance the question is an owner superhost?
The problem is how can we model the price of an accommodation with respect to the intrinsic features , extrinsic features and interaction features. So the objective of this project is finding the best model for the price variable which can be reduced to finding the most important and relevant intrinsic, extrinsic and interaction features. There are variables which are not appeared in dataset and we should invent them with respect to the context of the problem we can call them hidden variables which is should be unearthen.
```python
class AirbnbModel(keras.Model):
    def __init__(self):
        super(AirbnbModel, self).__init__()
        self.dense1 = layers.Dense(1028)
        self.dense2 = layers.Dense(512)
        self.dense3 = layers.Dense(1028)
        self.dense4 = layers.Dense(64)
        self.dense5 = layers.Dense(1)
        self.bn = BatchNormalization()
        self.drop1 = Dropout(0.05)
        self.drop2 = Dropout(0.05)
 
    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        x = self.drop2(x)
        x = tf.nn.relu(self.dense2(x))
        x = self.drop2(x)
        x = tf.nn.relu(self.dense3(x))
        x = self.drop2(x)
        x = tf.nn.relu(self.dense4(x))
        x = self.drop2(x)
        return tf.nn.relu(self.dense5(x))
model = AirbnbModel()
def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, residual/total)
    return r2
model.compile(optimizer = keras.optimizers.Adam(),loss = keras.losses.mean_squared_error,metrics = [R_squared])
model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 150, batch_size = 128, callbacks = [callbacks])
``` 
```python
 
1944/1944 [==============================] - 68s 35ms/step - loss: 0.0040 - R_squared: 0.9847 - val_loss: 0.0021 - val_R_squared: 0.9918 
```  
