#===========================================================================
#NN
# Keras
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
# 准备
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1)
FEATURE_COLS = numeric_cols + categorical_cols
TARGET_COL = 'Churn'
EPOCHS = 500
BATCH_SIZE = 100000
CLASS_WEIGHTS = {0 : 1., 1 : 2.5}
# Placeholders for the model input and embedding layers
cat_inputs = []
num_inputs = []
embeddings = []
embedding_layer_names = []
emb_n = 10

# Embedding for categorical features
for col in categorical_cols:
    _input = layers.Input(shape=[1], name=col)
    _embed = layers.Embedding(telcom[col].max() + 1, emb_n, name=col+'_emb')(_input)
    cat_inputs.append(_input)
    embeddings.append(_embed)
    embedding_layer_names.append(col+'_emb')
    
# Simple inputs for the numeric features          ##所有的数值特征，包含pca得到的新数值数据
for col in numeric_cols:
    numeric_input = layers.Input(shape=(1,), name=col)
    num_inputs.append(numeric_input)
    
# Merge the numeric inputs
merged_num_inputs = layers.concatenate(num_inputs)
#numeric_dense = layers.Dense(20, activation='relu')(merged_num_inputs)

# Merge embedding and use a Droput to prevent overfittting
merged_inputs = layers.concatenate(embeddings)
spatial_dropout = layers.SpatialDropout1D(0.2)(merged_inputs)
flat_embed = layers.Flatten()(spatial_dropout)

# Merge embedding and numeric features
all_features = layers.concatenate([flat_embed, merged_num_inputs])

# MLP for classification
x = layers.Dropout(0.2)(layers.Dense(100, activation='relu')(all_features))
x = layers.Dropout(0.2)(layers.Dense(50, activation='relu')(x))
x = layers.Dropout(0.2)(layers.Dense(25, activation='relu')(x))
x = layers.Dropout(0.2)(layers.Dense(15, activation='relu')(x))

# Final model
output = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=cat_inputs + num_inputs, outputs=output)
# Compile model with all parameters
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Definition model callbacks
# TB Callback
log_folder = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
tb_callback = callbacks.TensorBoard(
    log_dir=os.path.join('tb-logs', log_folder),
)

# Best model callback
bm_callback = callbacks.ModelCheckpoint(
    filepath=os.path.join('tb-logs', log_folder, 'bm.h5'),
    save_best_only=True,
    save_weights_only=False
)
# Training
##  class_weight 用sklearn 自动配置的方法1
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

_hist = model.fit(
    x=get_keras_dataset(train_df[FEATURE_COLS]),
    y=train_df[TARGET_COL],
    validation_data=(get_keras_dataset(test_df[FEATURE_COLS]), test_df[TARGET_COL]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=CLASS_WEIGHTS,
    callbacks=[tb_callback, bm_callback],
    verbose=2
)

# 绘制
plot_history(_hist)
# Evaluation
model = keras.models.load_model(os.path.join('tb-logs', log_folder, 'bm.h5'), compile=False)
pred = np.around(model.predict(get_keras_dataset(test_df[FEATURE_COLS])))

print(accuracy_score(test_df[TARGET_COL], pred))
print(classification_report(test_df[TARGET_COL], pred))