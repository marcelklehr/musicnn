from tensorflow.keras.models import Model
import tensorflow as tf

tags = ['Rock', 'Rap', 'Latin', 'Jazz', 'Electronic', 'Pop', 'Metal', 'RnB', 'Country', 'Reggae', 'Blues', 'Folk', 'Punk', 'World', 'New Age']
num_tags = len(tags)

num_filt = 1.6
inputs = layers.Input(shape=(188, mel_bins, 1))
X = inputs
normalized_input = layers.BatchNormalization()(X)

# tempo block
X128 = layers.BatchNormalization()(layers.Conv2D(num_filt*32, (128, 1), activation='relu', padding='same')(normalized_input))
X128P = layers.MaxPool2D(pool_size=(1, X128.shape[2]))(X128)
X64 = layers.BatchNormalization()(layers.Conv2D(num_filt*32, (64, 1), activation='relu', padding='same')(normalized_input))
X64P = layers.MaxPool2D(pool_size=(1,X64.shape[2]))(X64)
X32 = layers.BatchNormalization()(layers.Conv2D(num_filt*32, (32,1), activation='relu', padding='same')(normalized_input))
X32P = layers.MaxPool2D(pool_size=(1, X32.shape[2]))(X32)

# timbral block
padded_input = layers.ZeroPadding2D((3,0))(normalized_input)
X74 = layers.BatchNormalization()(layers.Conv2D(num_filt*128, (7, int(mel_bins * 0.4)), activation='relu', padding='valid')(padded_input))
X74P = layers.MaxPool2D(pool_size=(1, X74.shape[2]))(X74)

X77 = layers.BatchNormalization()(layers.Conv2D(num_filt*128, (7, int(mel_bins * 0.7)), activation='relu', padding='valid')(padded_input))
X77P = layers.MaxPool2D(pool_size=(1, X77.shape[2]))(X77)

print([X128P.shape, X64P.shape, X32P.shape, X74P.shape, X77P.shape])

frontend = layers.Permute([1,3,2])(layers.Concatenate(3)([X128P, X64P, X32P, X74P, X77P]))
X = frontend

# middle
num_filt = 512

C1 = layers.ZeroPadding2D((3,0))(X)
C1 = layers.BatchNormalization()(layers.Conv2D(filters=num_filt, kernel_size=[7, C1.shape[2]], activation='relu')(C1))
C1p = layers.Permute([1,3,2])(C1)
print(C1.shape)

C2 = layers.ZeroPadding2D((3,0))(C1)
C2 = layers.BatchNormalization()(layers.Conv2D(filters=num_filt, kernel_size=[7, C2.shape[2]], activation='relu')(C2))
print(C2.shape)
C2res = layers.Permute([1,3,2])(layers.Add()([C1, C2]))

C3 = layers.ZeroPadding2D((3,0))(C2)
C3 = layers.BatchNormalization()(layers.Conv2D(filters=num_filt, kernel_size=[7, C3.shape[2]], activation='relu')(C3))
C2res = layers.Permute([1,3,2])(layers.Add()([C2, C3]))

print(frontend.shape, C1.shape, C2res.shape, C2res.shape)

middle = layers.Concatenate(2)([frontend, C1p, C2res, C2res])
print(middle.shape)
X = layers.Concatenate(2)([
  layers.MaxPool2D(pool_size=(middle.shape[1], 1))(middle),
  layers.AvgPool2D(pool_size=(middle.shape[1], 1))(middle),
])
X = layers.Flatten()(X)
X = layers.BatchNormalization()(X)
X = layers.Dropout(0.75)(X)
X = layers.Dense(512, activation='relu')(X)
X = layers.BatchNormalization()(X)
X = layers.Dropout(0.75)(X)
predictions = layers.Dense(num_tags)(X)

model = Model(inputs=inputs, outputs=predictions)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.TopKCategoricalAccuracy(k=5)],
)
