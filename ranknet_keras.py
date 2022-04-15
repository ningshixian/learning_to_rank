import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Subtract, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


"""https://zhuanlan.zhihu.com/p/66497129"""


# H1、H2、H3、H4为四个使用RELU作为激活函数的全连接层
H1 = Dense(512, activation="relu")
H2 = Dense(256, activation="relu")
H3 = Dense(128, activation="relu")
H4 = Dense(1)

# 两个输入源
input_l = Input(shape=(NUM_FEATURES,))
input_r = Input(shape=(NUM_FEATURES,))

# Relevant document score.
h1_l = H1(input_l)
h2_l = H2(h1_l)
h3_l = H3(h2_l)
h4_l = H4(h3_l)

# Irrelevant document score.
h1_r = H1(input_r)
h2_r = H2(h1_r)
h3_r = H3(h2_r)
h4_r = H4(h3_r)

# 对两个encoder的输出进行求差操作
diff = Subtract()([h4_l, h4_r])

# 进行sigmoid激活
prob = Activation("sigmoid")(diff)

# 将各个模块进行组装
ranknet = Model(inputs=[input_l, input_r], outputs=prob)

# 设定激活函数
optimizer = Adam()

# 模型编译
ranknet.compile(optimizer=optimizer, loss="binary_crossentropy")

# 绘制模型结构图
plot_model(ranknet, to_file="ranknet.png", show_shapes=True, show_layer_names=True)


NUM_FEATURES = 100
NUM_SAMPLES = 50
BATCH_SIZE = 512
NUM_EPOCHS = 10

# 特征
X_l = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
X_r = np.random.rand(NUM_SAMPLES, NUM_FEATURES)
# 生成标签
y = (np.random.rand(NUM_SAMPLES) > 0.7).astype(int)
# Train model.
history = ranknet.fit([X_l, X_r], y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

plt.figure(figsize=(12, 7))
plt.plot(
    range(len(history.history.get("loss"))), history.history.get("loss"), linewidth=2.5
)
plt.grid(True)
plt.show()

# Generate scores from document/query features.
get_score = K.function([h1_l], [h4_l])
get_score([X_l])
get_score([X_r])