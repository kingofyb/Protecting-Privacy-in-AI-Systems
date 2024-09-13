import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Customise DPSGD
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

# MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# CNN structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Parameters
noise_levels = [0.1, 0.5, 1.0]  # Different noise level
results = {
    "noise_multiplier": [],
    "training_time": [],
    "accuracy": [],
    "response_time": [],
    "privacy_loss": []
}

for noise_multiplier in noise_levels:
    optimizer = DPSGD(learning_rate=0.01, l2_norm_clip=1.0, noise_multiplier=noise_multiplier)
    
    # record trainning time
    start_time = time.time()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=256, verbose=0)
    training_time = time.time() - start_time
    
    # record model accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Test single response time
    start_time = time.time()
    _ = model.predict(X_test[:1])
    response_time = time.time() - start_time
    
    # Assumed Privacy Loss
    privacy_loss = noise_multiplier  
    
    # save results
    results["noise_multiplier"].append(noise_multiplier)
    results["training_time"].append(training_time)
    results["accuracy"].append(test_acc)
    results["response_time"].append(response_time)
    results["privacy_loss"].append(privacy_loss)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
# plot graphs
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
