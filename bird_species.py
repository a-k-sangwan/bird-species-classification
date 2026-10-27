# %% [markdown]
# # Bird species classification with CNN, MobileNet and VGG16
#
# Data: 525-species bird dataset (CC0), Hugging Face copy
# chriamue/bird-species-dataset of the Kaggle gpiosenka/100-bird-species set.

# %%
import os

CONFIG = {
    "hf_dataset": "chriamue/bird-species-dataset",
    "data_dir": "/content/birds" if os.path.isdir("/content") else "birds",
    "n_classes": None,   # None = all 525, or an int for a random subset
    "img_size": 224,
    "batch_size": 64,
    "seed": 42,          # species subset
    "train_seeds": [42, 43, 44],
    "augment": True,
    "mixed_precision": True,
    "models": ["cnn", "mobilenet", "vgg16", "vgg16_ft"],
    "epochs": {"cnn": 80, "mobilenet": 50, "vgg16": 40, "vgg16_ft": 40},
    "optimizer": {"cnn": ["adamax", 0.006], "mobilenet": ["adamax", 0.006],
                  "vgg16": ["sgd", 0.01], "vgg16_ft": ["sgd", 0.01]},
    "vgg_head": "original",   # or "compact"
    "label_smoothing": 0.1,
    "warmup_epochs": 2,
    "tta": True,              # average with horizontally flipped test image
    "early_stop_patience": 8,
    "lr_patience": 3,
    "lr_factor": 0.5,
}
OUT_DIR = "/content/drive/MyDrive/BM616_birds"

# %%
import glob
import json
import random
import time

import numpy as np
import pandas as pd

# overrides for runs from the command line
CONFIG.update(json.loads(os.environ.get("BM616_CONFIG", "{}")))
OUT_DIR = os.environ.get("BM616_OUT_DIR", OUT_DIR)

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    OUT_DIR = os.environ.get("BM616_OUT_DIR", os.path.abspath("BM616_birds"))

USE_DRIVE = IN_COLAB and os.environ.get("BM616_DRIVE", "1") != "0"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

if IN_COLAB:
    # fonts for the figures
    os.system("echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula "
              "select true | debconf-set-selections")
    os.system("apt-get -qq install -y ttf-mscorefonts-installer fonts-stix > /dev/null")

import tensorflow as tf
import keras
from keras import layers

RESULTS_DIR = os.path.join(OUT_DIR, "results")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUT_DIR, "figures")
for d in (RESULTS_DIR, CKPT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

keras.utils.set_random_seed(CONFIG["seed"])
if CONFIG["mixed_precision"]:
    keras.mixed_precision.set_global_policy("mixed_float16")

print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# %%
# download and unpack the data
import tarfile

from huggingface_hub import hf_hub_download

data_root = CONFIG["data_dir"]
for split in ("train", "valid", "test"):
    marker = os.path.join(data_root, f".{split}_unpacked")
    if os.path.exists(marker):
        continue
    archive = hf_hub_download(CONFIG["hf_dataset"], f"data/{split}.tar.gz",
                              repo_type="dataset")
    print("unpacking", split)
    with tarfile.open(archive) as tar:
        tar.extractall(data_root, filter="data")
    open(marker, "w").close()


def find_split_root(root):
    for dirpath, dirnames, _ in os.walk(root):
        names = {d.lower() for d in dirnames}
        if {"train", "valid", "test"} <= names:
            return dirpath
    raise FileNotFoundError(f"no train/valid/test folders under {root}")


split_root = find_split_root(data_root)
print("data in", split_root)


def norm(name):
    # some species names differ between splits by case or spacing
    return " ".join(name.upper().split())


def list_split(split):
    split_dir = next(os.path.join(split_root, d) for d in os.listdir(split_root)
                     if d.lower() == split)
    files = {}
    for cls in os.listdir(split_dir):
        cdir = os.path.join(split_dir, cls)
        if not os.path.isdir(cdir):
            continue
        imgs = sorted(glob.glob(os.path.join(cdir, "*")))
        imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.setdefault(norm(cls), []).extend(imgs)
    return files


raw = {s: list_split(s) for s in ("train", "valid", "test")}
common = sorted(set(raw["train"]) & set(raw["valid"]) & set(raw["test"]))
dropped = sorted((set(raw["train"]) | set(raw["valid"]) | set(raw["test"])) - set(common))
if dropped:
    print("species missing from a split, dropped:", dropped)

if CONFIG["n_classes"]:
    rng = random.Random(CONFIG["seed"])
    common = sorted(rng.sample(common, CONFIG["n_classes"]))

CLASS_NAMES = common
NUM_CLASSES = len(CLASS_NAMES)
label_of = {c: i for i, c in enumerate(CLASS_NAMES)}


def items(split):
    return [(p, label_of[c]) for c in CLASS_NAMES for p in raw[split][c]]


TRAIN, VALID, TEST = items("train"), items("valid"), items("test")
print(f"{NUM_CLASSES} species | train {len(TRAIN)} | valid {len(VALID)} | test {len(TEST)}")

with open(os.path.join(RESULTS_DIR, "classes.json"), "w") as f:
    json.dump(CLASS_NAMES, f, indent=1)
counts = pd.DataFrame({
    "species": CLASS_NAMES,
    "train": [len(raw["train"][c]) for c in CLASS_NAMES],
    "valid": [len(raw["valid"][c]) for c in CLASS_NAMES],
    "test": [len(raw["test"][c]) for c in CLASS_NAMES],
})
counts.to_csv(os.path.join(RESULTS_DIR, "class_counts.csv"), index=False)
print(counts[["train", "valid", "test"]].describe().loc[["min", "50%", "max"]])

# %%
# input pipeline; pixels stay in 0-255 and each model preprocesses itself
IMG = CONFIG["img_size"]
AUTOTUNE = tf.data.AUTOTUNE


def decode(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                             expand_animations=False)
    img.set_shape([None, None, 3])
    return img, label


def resize(img, label):
    return tf.image.resize(img, [IMG, IMG]), label


def cached(split_items):
    paths = [p for p, _ in split_items]
    labels = [l for _, l in split_items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(decode, num_parallel_calls=AUTOTUNE).cache()


def make_ds(base, seed=None):
    ds = base
    if seed is not None:   # shuffle only the training set
        ds = ds.shuffle(len(TRAIN), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(resize, num_parallel_calls=AUTOTUNE)
    return ds.batch(CONFIG["batch_size"]).prefetch(AUTOTUNE)


train_base = cached(TRAIN)
valid_ds = make_ds(cached(VALID))
test_ds = make_ds(cached(TEST))

# %%
# models


def augmentation():
    # no vertical flip, test birds are never upside down
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name="augment")


class VGGPreprocess(layers.Layer):
    # same as keras.applications.vgg16.preprocess_input

    def call(self, x):
        x = x[..., ::-1]
        return x - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)


def classifier(x):
    # float32 softmax for mixed precision
    return layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32",
                        name="predictions")(x)


def build_cnn():
    inp = keras.Input((IMG, IMG, 3))
    x = augmentation()(inp) if CONFIG["augment"] else inp
    x = layers.Rescaling(1 / 255.0)(x)
    for filters, n_conv in [(8, 2), (16, 2), (32, 2), (64, 2), (128, 2),
                            (256, 2), (512, 1)]:
        for _ in range(n_conv):
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inp, classifier(x), name="cnn")


def build_mobilenet():
    inp = keras.Input((IMG, IMG, 3))
    x = augmentation()(inp) if CONFIG["augment"] else inp
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    base = keras.applications.MobileNet(include_top=False, weights="imagenet",
                                        input_shape=(IMG, IMG, 3))
    x = base(x)
    x = layers.MaxPooling2D(x.shape[1])(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    return keras.Model(inp, classifier(x), name="mobilenet")


def build_vgg16(fine_tune=False):
    inp = keras.Input((IMG, IMG, 3))
    x = augmentation()(inp) if CONFIG["augment"] else inp
    x = VGGPreprocess()(x)
    base = keras.applications.VGG16(include_top=False, weights="imagenet",
                                    input_shape=(IMG, IMG, 3))
    base.trainable = False
    if fine_tune:   # train block 5 only
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")
    x = base(x, training=False)
    if CONFIG["vgg_head"] == "original":
        x = layers.Flatten()(x)
        for units, drop in [(10000, 0.5), (10000, 0.4)]:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(drop)(x)
        x = layers.Dense(4000, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(700, activation="relu")(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)
    return keras.Model(inp, classifier(x), name="vgg16")


BUILDERS = {"cnn": build_cnn, "mobilenet": build_mobilenet, "vgg16": build_vgg16,
            "vgg16_ft": lambda: build_vgg16(fine_tune=True)}


def run_id(name, seed):
    return f"{name}_s{seed}"


def smoothed_sparse_ce(eps):
    # keras has label smoothing only for one-hot labels
    def loss(y_true, y_pred):
        y = keras.ops.one_hot(keras.ops.cast(keras.ops.reshape(y_true, (-1,)),
                                             "int32"), NUM_CLASSES)
        return keras.losses.categorical_crossentropy(y, y_pred,
                                                     label_smoothing=eps)
    return loss


def compile_model(model, name):
    kind, lr = CONFIG["optimizer"][name]
    opt = {"adam": keras.optimizers.Adam, "adamax": keras.optimizers.Adamax,
           "sgd": keras.optimizers.SGD}[kind](lr)
    model.compile(
        optimizer=opt,
        loss=(smoothed_sparse_ce(CONFIG["label_smoothing"])
              if CONFIG["label_smoothing"] else "sparse_categorical_crossentropy"),
        metrics=["accuracy",
                 keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5")],
    )
    return model


# %%
# training


class Warmup(keras.callbacks.Callback):
    # linear lr ramp over the first few epochs

    def __init__(self, base_lr, epochs, steps_per_epoch):
        super().__init__()
        self.base_lr, self.total = base_lr, epochs * steps_per_epoch
        self.steps, self.epoch = steps_per_epoch, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        step = self.epoch * self.steps + batch + 1
        if step <= self.total:
            self.model.optimizer.learning_rate = self.base_lr * step / self.total


class EpochInfo(keras.callbacks.Callback):
    # log epoch time and lr

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_time_s"] = time.time() - self.t0
        logs["lr"] = float(keras.ops.convert_to_numpy(
            self.model.optimizer.learning_rate))


def train(name, seed):
    run = run_id(name, seed)
    done_flag = os.path.join(RESULTS_DIR, f"{run}_trained.json")
    if os.path.exists(done_flag):
        print(f"{run}: already trained, skipping")
        return
    backup = os.path.join(CKPT_DIR, f"{run}_backup")
    log_path = os.path.join(RESULTS_DIR, f"{run}_log.csv")
    resuming = os.path.isdir(backup) and os.listdir(backup)
    if not resuming and os.path.exists(log_path):
        os.remove(log_path)

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = compile_model(BUILDERS[name](), name)
    print(f"{run}: {model.count_params() / 1e6:.2f}M parameters"
          + (" (resuming)" if resuming else ""))

    callbacks = [
        keras.callbacks.BackupAndRestore(backup),
        EpochInfo(),
        *([Warmup(CONFIG["optimizer"][name][1], CONFIG["warmup_epochs"],
                  -(-len(TRAIN) // CONFIG["batch_size"]))]
          if CONFIG["warmup_epochs"] else []),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                          factor=CONFIG["lr_factor"],
                                          patience=CONFIG["lr_patience"],
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      patience=CONFIG["early_stop_patience"],
                                      verbose=1),
        keras.callbacks.ModelCheckpoint(
            os.path.join(CKPT_DIR, f"{run}_best.weights.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(log_path, append=True),
    ]
    model.fit(make_ds(train_base, seed), validation_data=valid_ds,
              epochs=CONFIG["epochs"][name], callbacks=callbacks,
              verbose=1 if USE_DRIVE else 2)

    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    with open(done_flag, "w") as f:
        json.dump({"params_total": int(model.count_params()),
                   "params_trainable": trainable}, f)


# %%
# evaluation on the test set
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, top_k_accuracy_score)


def evaluate(name, seed):
    run = run_id(name, seed)
    if os.path.exists(os.path.join(RESULTS_DIR, f"{run}_eval.json")):
        return
    model = compile_model(BUILDERS[name](), name)
    model.load_weights(os.path.join(CKPT_DIR, f"{run}_best.weights.h5"))

    y_true = np.array([l for _, l in TEST])
    t0 = time.time()
    prob = model.predict(test_ds, verbose=0)
    ms_per_img = 1000 * (time.time() - t0) / len(y_true)
    acc_single = float(accuracy_score(y_true, prob.argmax(1)))
    if CONFIG["tta"]:
        flipped = test_ds.map(lambda x, y: (tf.image.flip_left_right(x), y))
        prob = (prob + model.predict(flipped, verbose=0)) / 2
    y_pred = prob.argmax(1)

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    pd.DataFrame({"species": CLASS_NAMES, "precision": p, "recall": r,
                  "f1": f1, "support": sup}).to_csv(
        os.path.join(RESULTS_DIR, f"{run}_per_class.csv"), index=False)
    np.save(os.path.join(RESULTS_DIR, f"{run}_confusion.npy"),
            confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)))
    np.save(os.path.join(RESULTS_DIR, f"{run}_test_prob.npy"),
            prob.astype(np.float16))

    log = pd.read_csv(os.path.join(RESULTS_DIR, f"{run}_log.csv"))
    best = int(log["val_loss"].idxmin())
    with open(os.path.join(RESULTS_DIR, f"{run}_trained.json")) as f:
        params = json.load(f)
    result = {
        "model": name,
        "seed": seed,
        "n_classes": NUM_CLASSES,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_no_tta": acc_single,
        "top5_accuracy": float(top_k_accuracy_score(
            y_true, prob, k=5, labels=range(NUM_CLASSES))),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1.mean()),
        "epochs_run": int(len(log)),
        "best_epoch": best + 1,
        "best_train_accuracy": float(log["accuracy"][best]),
        "best_val_accuracy": float(log["val_accuracy"][best]),
        "train_minutes": float(log["epoch_time_s"].sum() / 60),
        "test_ms_per_image": ms_per_img,
        **params,
        "config": CONFIG,
    }
    with open(os.path.join(RESULTS_DIR, f"{run}_eval.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"{run}: acc {result['accuracy']:.4f} | macro F1 {result['macro_f1']:.4f}"
          f" | {result['epochs_run']} epochs, {result['train_minutes']:.0f} min")
    return result


# %%
for seed in CONFIG["train_seeds"]:
    for name in CONFIG["models"]:
        train(name, seed)
        evaluate(name, seed)

summary = pd.DataFrame([
    json.load(open(os.path.join(RESULTS_DIR, f"{run_id(m, s)}_eval.json")))
    for s in CONFIG["train_seeds"] for m in CONFIG["models"]]).drop(columns="config")
print(summary.to_string())
print(summary.groupby("model")[["accuracy", "top5_accuracy", "macro_f1",
                                "epochs_run", "train_minutes"]].agg(["mean", "std"]))

# %%
# [plots.py]
from plots import *

# %%
# figures and tables
set_style()
fig_samples(TRAIN, CLASS_NAMES, FIG_DIR)
fig_class_counts(counts["train"], FIG_DIR)
make_all(RESULTS_DIR, FIG_DIR, CONFIG["models"], CONFIG["train_seeds"],
         CLASS_NAMES, TRAIN)

if USE_DRIVE:
    from IPython.display import Image as ShowImage, display
    for png in sorted(glob.glob(os.path.join(FIG_DIR, "*.png"))):
        print(os.path.basename(png))
        display(ShowImage(png, width=700))

# %%
# zip results and figures
import zipfile

zip_path = os.path.join(OUT_DIR, "bm616_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("results", "figures"):
        for path in glob.glob(os.path.join(OUT_DIR, sub, "*")):
            if not path.endswith("_test_prob.npy"):
                z.write(path, os.path.relpath(path, OUT_DIR))
print(zip_path)
if USE_DRIVE:
    from google.colab import files
    files.download(zip_path)
