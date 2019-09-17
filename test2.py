import tensorflow as tf
import numpy as np

input = tf.keras.Input(shape=(5,), name='input')
allowed_moves = tf.keras.Input(shape=(2,), name='allow')
middle = tf.keras.layers.Dense(10, activation=tf.nn.tanh, name='middle')(input)
out1 = tf.keras.layers.Dense(2, activation=tf.nn.tanh, name='out1')(middle)
out2 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='out2')(middle)
multiply = tf.keras.layers.Multiply()([allowed_moves, out1])

model = tf.keras.Model(inputs=[input, allowed_moves],
                    outputs=[multiply, out2])

model.compile(optimizer='adam',
              loss=[tf.compat.v2.losses.mean_squared_error, tf.compat.v2.losses.mean_squared_error],
              metrics=['accuracy'])

class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        print(model.predict_on_batch({'input': np.array([[1,0,1,0,1]]), 'allow': np.array([[1.0, 1.0]])})[0].numpy())
        print(model.predict_on_batch({'input': np.array([[1,0,1,0,1]]), 'allow': np.array([[1.0, 1.0]])})[1].numpy())
        return {'input': np.array([[1,0,1,0,1],[1,1,1,0,1]]), 'allow': np.array([[1.0, 1.0], [1.0, 1.0]])}, {'multiply': np.array([[1.0, -1.0], [1.0, 1.0]]), 'out2': np.array([[-1.0], [1.0]])}

generator = GameStateGenerator()

model.fit_generator(generator=generator, epochs=1, workers=1)

print(model.summary())
print(model.predict({'input': np.array([[1,0,1,0,1]]), 'allow': np.array([[0.0, 0.0]])}))
print(model.predict({'input': np.array([[1,1,1,0,1]]), 'allow': np.array([[0.0, 1.0]])}))