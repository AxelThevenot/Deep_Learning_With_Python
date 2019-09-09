from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.models import Model

# region load false data
vocabulary_size = 50000
n_income_group = 10
# endregion


# region model
posts_input = Input(shape=(None, ), dtype='int32', name='posts')
posts_embedded = Embedding(256, vocabulary_size)(posts_input)
x = Conv1D(128, 5, activation='relu')(posts_embedded)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)

# region predictions
age_pred = Dense(1, name='age')(x)
income_pred = Dense(1, name='income')(x)
gender_pred = Dense(1, activation='sigmoid', name='gender')(x)
# endregion

model = Model(posts_input, [age_pred, income_pred, gender_pred])
model.compile(loss=        ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weigths=[0.25 , 1.                        , 10.                  ])

# endregion


posts = age_targets = income_targets = gender_targets = None

model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10,
          batch_size=64)

# or
model.fit(posts, {'age':age_targets,
                  'income':income_targets,
                  'gender':gender_targets},  # work if names are given
          epochs=10,
          batch_size=64)
