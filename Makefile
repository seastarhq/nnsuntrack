# Sun Tracker CNN pipeline.
#
#   make data       generate the synthetic dataset
#   make train      train the CNN, export .keras + .tflite
#   make metrics    latency / accuracy report for the .tflite
#   make visualize  centroid grid PNG
#   make infer      single-image inference
#   make smoke      fast end-to-end run in .smoke/ (leaves real artifacts alone)
#
# Stages are wired by file dependency, so `make metrics` builds the dataset and
# the model first if they are missing or out of date. Changing a parameter
# (NUM_IMAGES, EPOCHS, ...) does *not* invalidate an existing artifact - use
# `make -B <target>` to force a rebuild.

# tee must not swallow a script's failure.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

# Two interpreters, because TensorFlow has no wheels for Python 3.14:
#   PY       3.14 venv - data generation and the ai-edge-litert runtime path
#                        (metrics/visualize/infer), i.e. what the RPi5 runs.
#   TRAIN_PY 3.13 venv - TensorFlow, for training and .tflite conversion only.
PY       ?= .venv/bin/python
TRAIN_PY ?= .venv313/bin/python

DATA_DIR   ?= synthetic_images
NUM_IMAGES ?= 20000
DATA_ARGS  ?=

# TF 2.21's RPATH misses the pip-installed libcusolver.so.11, so it drops to
# CPU (and then errors out during TFLite conversion rather than falling back
# cleanly). Putting the nvidia/*/lib dirs on LD_LIBRARY_PATH fixes it.
# Evaluated lazily - only the training rule expands this.
NVIDIA_LIB_PATH = $(shell $(TRAIN_PY) -c "import glob, os, nvidia; \
    print(':'.join(sorted(glob.glob(os.path.join(os.path.dirname(nvidia.__file__), '*', 'lib')))))" 2>/dev/null)

# To force CPU training instead: make train TRAIN_ENV=CUDA_VISIBLE_DEVICES=-1
TRAIN_ENV ?= LD_LIBRARY_PATH="$(NVIDIA_LIB_PATH):$$LD_LIBRARY_PATH"

MODEL             ?= sun_detector_model
EPOCHS            ?= 50
BATCH_SIZE        ?= 32
VALIDATION_SPLIT  ?= 0.2
TRAIN_ARGS        ?=

CONFIDENCE_THRESHOLD ?= 0.5
NUM_THREADS          ?= 4

IMAGE   ?= $(DATA_DIR)/image_0.png
VIZ_OUT ?= centroids_grid.png
LOG_DIR ?= logs

SMOKE_DIR ?= .smoke

KERAS    := $(MODEL).keras
TFLITE   := $(MODEL).tflite
WEIGHTS  := $(MODEL)_best.weights.h5
METADATA := $(DATA_DIR)/metadata.csv

# A relative/absolute interpreter path that does not exist is almost always a
# missing venv; fail with something more useful than "No such file".
ifneq (,$(findstring /,$(PY)))
ifeq (,$(wildcard $(PY)))
$(error Python interpreter '$(PY)' not found. Create the venv, or override: make PY=python3 ...)
endif
endif

# TRAIN_PY is only needed by the training rule, so check it there rather than
# blocking every target when the TensorFlow venv is absent.
check-train-py:
	@test -x '$(TRAIN_PY)' || { \
	    echo "TensorFlow interpreter '$(TRAIN_PY)' not found."; \
	    echo "Training needs Python <=3.13 (no TF wheels for 3.14). Recreate with:"; \
	    echo "  uv venv --python 3.13 .venv313"; \
	    echo "  uv pip install --python .venv313 tensorflow scikit-learn opencv-python pandas matplotlib"; \
	    exit 1; }

.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

# ---------------------------------------------------------------- pipeline ---

## data: generate synthetic images + metadata.csv
data: $(METADATA)

$(METADATA): synthetic_data_script.py | $(LOG_DIR)
	$(PY) synthetic_data_script.py \
	    --num_images $(NUM_IMAGES) \
	    --output_dir $(DATA_DIR) \
	    $(DATA_ARGS) 2>&1 | tee $(LOG_DIR)/data.log

## train: train the CNN and export .keras + .tflite
train: $(TFLITE)

# One invocation produces both models, so declare them as a grouped target
# (GNU Make 4.3+); otherwise a parallel build would run the training twice.
$(KERAS) $(TFLITE) &: training_script.py $(METADATA) | check-train-py $(LOG_DIR)
	$(TRAIN_ENV) $(TRAIN_PY) training_script.py \
	    --data_dir $(DATA_DIR) \
	    --epochs $(EPOCHS) \
	    --batch_size $(BATCH_SIZE) \
	    --validation_split $(VALIDATION_SPLIT) \
	    --output_model $(KERAS) \
	    --output_tflite $(TFLITE) \
	    $(TRAIN_ARGS) 2>&1 | tee $(LOG_DIR)/train.log

## metrics: latency + accuracy report for the TFLite model
metrics: $(TFLITE) $(METADATA) | $(LOG_DIR)
	$(PY) metrics_script.py \
	    --data_dir $(DATA_DIR) \
	    --model_path $(TFLITE) \
	    --confidence_threshold $(CONFIDENCE_THRESHOLD) \
	    --num_threads $(NUM_THREADS) 2>&1 | tee $(LOG_DIR)/metrics.log

## visualize: render a grid of predicted centroids (VIZ_OUT=centroids_grid.png)
visualize: $(TFLITE) $(METADATA) | $(LOG_DIR)
	$(PY) visualize_script.py \
	    --data_dir $(DATA_DIR) \
	    --model_path $(TFLITE) \
	    --confidence_threshold $(CONFIDENCE_THRESHOLD) \
	    --num_threads $(NUM_THREADS) \
	    --show_truth \
	    --output $(VIZ_OUT) 2>&1 | tee $(LOG_DIR)/visualize.log

## infer: run inference on one image (override with IMAGE=path/to.png)
infer: $(TFLITE)
	$(PY) deployment_script.py \
	    --image_path $(IMAGE) \
	    --model_path $(TFLITE) \
	    --confidence_threshold $(CONFIDENCE_THRESHOLD) \
	    --num_threads $(NUM_THREADS)

## smoke: 400 images / 5 epochs end-to-end, contained in .smoke/
smoke:
	$(MAKE) --no-print-directory \
	    DATA_DIR=$(SMOKE_DIR)/synthetic_images \
	    MODEL=$(SMOKE_DIR)/sun_detector_model \
	    LOG_DIR=$(SMOKE_DIR)/logs \
	    NUM_IMAGES=400 EPOCHS=5 \
	    infer

$(LOG_DIR):
	mkdir -p $@

# ----------------------------------------------------------------- cleanup ---

## clean: remove models, logs, and the visualization PNG (keeps the dataset)
clean:
	rm -fv $(KERAS) $(TFLITE) $(WEIGHTS) $(VIZ_OUT)
	rm -rfv $(LOG_DIR) $(SMOKE_DIR)

## distclean: clean, and also delete the generated dataset directory
distclean: clean
	rm -rfv $(DATA_DIR)

## help: list targets
help:
	@echo 'Sun Tracker CNN pipeline. Targets:'
	@echo
	@sed -n 's/^## \(.*\)/  \1/p' $(MAKEFILE_LIST)
	@echo
	@echo 'Common overrides (current values):'
	@echo '  PY=$(PY)  (3.14: data + litert runtime)'
	@echo '  TRAIN_PY=$(TRAIN_PY)  (3.13: TensorFlow)'
	@echo '  DATA_DIR=$(DATA_DIR)  MODEL=$(MODEL)'
	@echo '  NUM_IMAGES=$(NUM_IMAGES)  EPOCHS=$(EPOCHS)  BATCH_SIZE=$(BATCH_SIZE)'
	@echo '  CONFIDENCE_THRESHOLD=$(CONFIDENCE_THRESHOLD)  NUM_THREADS=$(NUM_THREADS)'
	@echo
	@echo 'e.g.  make train NUM_IMAGES=1000 EPOCHS=10'
	@echo '      make -B data      # force regeneration'

.PHONY: data train metrics visualize infer smoke clean distclean help check-train-py
