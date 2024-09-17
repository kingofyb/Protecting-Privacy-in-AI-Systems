import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# customise DPSGD
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

# load MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full, X_test = X_train_full / 255.0, X_test / 255.0

# split dataset
X_train, X_shadow, y_train, y_shadow = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=42)
X_attack = np.concatenate((X_train[:100], X_shadow[:100]))
y_attack = np.concatenate((y_train[:100], y_shadow[:100]))

# CNN model structure
def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
    return model

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

# NO DP model
model_vanilla = create_model()
model_vanilla.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_vanilla.fit(X_train, y_train, epochs=5, batch_size=256, verbose=0)

# DP model
optimizer_dp = DPSGD(learning_rate=0.01, l2_norm_clip=1.0, noise_multiplier=0.1)
model_dp = create_model()
model_dp.compile(optimizer=optimizer_dp, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dp.fit(X_train, y_train, epochs=5, batch_size=256, verbose=0)

# attck
predictions_vanilla = model_vanilla.predict(X_attack)
predictions_dp = model_dp.predict(X_attack)

# compare
confidence_vanilla_train = np.max(predictions_vanilla[:100], axis=1)
confidence_vanilla_shadow = np.max(predictions_vanilla[100:], axis=1)

confidence_dp_train = np.max(predictions_dp[:100], axis=1)
confidence_dp_shadow = np.max(predictions_dp[100:], axis=1)

# plot graph
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(confidence_vanilla_train, bins=10, alpha=0.5, label='Train (No DP)')
plt.hist(confidence_vanilla_shadow, bins=10, alpha=0.5, label='Shadow (No DP)')
plt.title('Membership Inference Attack (No DP)')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(confidence_dp_train, bins=10, alpha=0.5, label='Train (DP)')
plt.hist(confidence_dp_shadow, bins=10, alpha=0.5, label='Shadow (DP)')
plt.title('Membership Inference Attack (DP)')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
