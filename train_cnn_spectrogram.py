"""Train the report-aligned CNN on processed_spectrograms/<split>/<label> folders."""

from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cnn_model.keras"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    data = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        image_size=(128, 128),
        batch_size=32,
        label_mode="binary",
        validation_split=0.2,
        subset="both",
        seed=42,
    )
    train_ds, validation_ds = data
    class_names = train_ds.class_names
    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization(x), y))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs)

    labels = []
    predictions = []
    for images, batch_labels in validation_ds:
        labels.extend(batch_labels.numpy().astype(int).ravel())
        predictions.extend((model.predict(images, verbose=0).ravel() >= 0.5).astype(int))
    print(classification_report(labels, predictions, target_names=class_names, zero_division=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved model to {args.output.resolve()}")


if __name__ == "__main__":
    main()
