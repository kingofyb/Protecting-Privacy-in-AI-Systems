import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# 自定义DPSGD优化器类
class DPSGD(tf.keras.optimizers.SGD):
    def __init__(self, l2_norm_clip=1.0, noise_multiplier=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier

    def _compute_gradients(self, tape, loss, var_list, grad_loss=None):
        gradients = tape.gradient(loss, var_list, grad_loss)
        clipped_grads = []
        for g in gradients:
            if g is not None:
                norm = tf.norm(g)
                clip_factor = self.l2_norm_clip / (norm + 1e-6)
                clipped_g = g * tf.minimum(clip_factor, 1.0)
                noise = tf.random.normal(shape=g.shape, stddev=self.l2_norm_clip * self.noise_multiplier)
                clipped_grads.append(clipped_g + noise)
            else:
                clipped_grads.append(g)
        return zip(clipped_grads, var_list)

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# 定义模型结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 测试参数
noise_levels = [0.1, 0.5, 1.0]  # 不同的噪声级别
results = {
    "noise_multiplier": [],
    "training_time": [],
    "accuracy": [],
    "response_time": [],
    "privacy_loss": []
}

for noise_multiplier in noise_levels:
    optimizer = DPSGD(learning_rate=0.01, l2_norm_clip=1.0, noise_multiplier=noise_multiplier)
    
    # 记录训练时间
    start_time = time.time()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=256, verbose=0)
    training_time = time.time() - start_time
    
    # 记录模型准确率
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # 测试单次预测的响应时间
    start_time = time.time()
    _ = model.predict(X_test[:1])
    response_time = time.time() - start_time
    
    # 假设的隐私损耗
    privacy_loss = noise_multiplier  # 这里只是一个简化的表示，实际中ε值计算更复杂
    
    # 保存结果
    results["noise_multiplier"].append(noise_multiplier)
    results["training_time"].append(training_time)
    results["accuracy"].append(test_acc)
    results["response_time"].append(response_time)
    results["privacy_loss"].append(privacy_loss)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
# 数据可视化
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(results["noise_multiplier"], results["privacy_loss"], marker='o')
plt.title('Privacy Loss vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Privacy Loss (Epsilon)')

plt.subplot(2, 2, 2)
plt.plot(results["noise_multiplier"], results["training_time"], marker='o')
plt.title('Training Time vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Training Time (seconds)')

plt.subplot(2, 2, 3)
plt.plot(results["noise_multiplier"], results["accuracy"], marker='o')
plt.title('Accuracy vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 4)
plt.plot(results["noise_multiplier"], results["response_time"], marker='o')
plt.title('Response Time vs Noise Multiplier')
plt.xlabel('Noise Multiplier')
plt.ylabel('Response Time (seconds)')

plt.tight_layout()
plt.show()
