import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_optimizer_class

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# 创建一个简单的模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 设置差分隐私优化器
dp_optimizer_class = make_keras_optimizer_class(tf.keras.optimizers.SGD)
optimizer = dp_optimizer_class(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=250,
    learning_rate=0.15
)

# 编译模型
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=1, batch_size=250)

# 测试模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
