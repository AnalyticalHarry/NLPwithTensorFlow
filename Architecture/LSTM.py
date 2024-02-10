from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# tokenize text data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(tweets_df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(tweets_df['cleaned_text'])
padded_sequences = pad_sequences(sequences, padding='post')

# word index
word_index = tokenizer.word_index

#Split data into train and text
train_texts, test_texts, train_labels, test_labels = train_test_split( df['cleaned_text'], 
                                                                       df['encode'], 
                                                                       test_size=0.3, 
                                                                       random_state=0)
                                                                       
(train_texts.shape, train_labels.shape), (test_texts.shape, test_labels.shape)

tokenizer = Tokenizer(num_words=10000) 
tokenizer.fit_on_texts(train_texts)

# texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
# pad the sequences to ensure uniform length
train_padded = pad_sequences(train_sequences, maxlen=100)  
test_padded = pad_sequences(test_sequences, maxlen=100)

#creating model 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])

#EarlyStopping and ModelCheckpoint callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3,  
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    'best_model.h5',  
    save_best_only=True, 
    monitor='val_loss', 
    verbose=1
)

#compile model 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model summary
model.summary()

train_labels = train_labels - 1
test_labels = test_labels - 1

history = model.fit(
    train_padded, 
    train_labels, 
    epochs=10, 
    batch_size=32, 
    validation_data=(test_padded, test_labels), 
    callbacks=[early_stopping, model_checkpoint]
)

test_loss, test_acc = model.evaluate(test_padded, test_labels)
print('Test Accuracy:', test_acc)

y_pred = model.predict(test_padded)

# training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid(True, ls='--', alpha=0.2, color='black')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid(True, ls='--', alpha=0.2, color='black')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

## Code created by Hemant Thapa
## Date: 10.02.2024
## Email: hemantthapa1998@gmail.com
                                                                
