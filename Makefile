SHELL               := /bin/bash

# see git describe documentation for a descption
# the version as set in meta.yaml
RELEASE_VERSION     = $(shell cat VERSION)
PROJECT_NAME        = $(shell grep -Eo "name: .*" conda-build/meta.yaml | cut -f 2 -d ' ')

GIT_URL             = $(shell git config --get remote.origin.url)
GIT_REV             = $(shell git rev-parse --short HEAD)
DEV_VERSION         = $(RELEASE_VERSION)_dev$(GIT_REV)

# conda-build 2.1.12 started breaking our builds.
#  2.1.10 was the last version we know works so let's lock to this and more carefully update.
CONDA_BUILD_VERSION = 2.1.10

CONDA_OUTPUT_FOLDER ?= /opt/releases/driver/
# where to put the build objects. We can't use the default because
# there is a known bug by which conda can't build on encrypted drives
CONDA_BUILD_FOLDER  = /tmp/$(PROJECT_NAME)_$(DEV_VERSION)_$(shell whoami)
# EXPORT_ALL_VARIABLES: will cause conflict with conda specific ENV variables.
# this is required to satisfy our dependencies
PYTHON_VERSION      = 3.5

DRIVER_CONDA_CHANNELS = -c conda -c driver

export GIT_LFS_SKIP_SMUDGE=1

.EXPORT_ALL_VARIABLES:

####################################################################################################
# Development commands
#

develop: install-deps-in-current-env
	VERSION=$(DEV_VERSION) DRIVER_BUILD_GIT_URL=$(GIT_URL) DRIVER_BUILD_GIT_REV=$(GIT_REV) \
	    python .package_install_steps.py develop

# builds a conda package from the meta.yaml file and places it into the local repo
build-in-current-env: verify-conda-build-installed
	-mkdir -p $(CONDA_OUTPUT_FOLDER) && test -w $(CONDA_OUTPUT_FOLDER)
	conda config --set anaconda_upload no
	VERSION=$(DEV_VERSION) DRIVER_BUILD_GIT_URL=$(GIT_URL) DRIVER_BUILD_GIT_REV=$(GIT_REV) \
	    conda-build conda-build/meta.yaml \
		$(DRIVER_CONDA_CHANNELS) \
		--croot $(CONDA_BUILD_FOLDER) \
		--output-folder $(CONDA_OUTPUT_FOLDER) \
		--python $(PYTHON_VERSION)

# Install package into the current environment
install-in-current-env: build-in-current-env
	conda install --yes $(DRIVER_CONDA_CHANNELS) -c file://$(CONDA_OUTPUT_FOLDER) $(PROJECT_NAME)=$(DEV_VERSION)

uninstall-from-current-env: # develop uninstall  # install uninstall
	-conda uninstall --yes $(PROJECT_NAME) || pip uninstall --yes $(PROJECT_NAME)

install-deps-in-current-env: install-in-current-env uninstall-from-current-env

####################################################################################################
# Conda environment commands
#

# run the test scripts in a clean conda environment
setup-clean-env:
	-conda create --yes --name test_env_$(PROJECT_NAME)_$(DEV_VERSION) python=$(PYTHON_VERSION)

teardown-clean-env:
	conda remove --name test_env_$(PROJECT_NAME)_$(DEV_VERSION) --all --yes

####################################################################################################
# Release commands
#

git-tag:
	git pull --tags
	git tag -a v$(RELEASE_VERSION) -m "v$(RELEASE_VERSION)"
	git push origin v$(RELEASE_VERSION)

build-release-and-upload: verify-conda-build-installed
	conda config --set anaconda_upload yes
	# Create the package if it does not already exist and set the permission to
	# private (to driver).
	-anaconda package driver/$(PROJECT_NAME) --create --private
	VERSION=$(RELEASE_VERSION) DRIVER_BUILD_GIT_URL=$(GIT_URL) DRIVER_BUILD_GIT_REV=$(GIT_REV) \
	    conda-build conda-build/meta.yaml \
		$(DRIVER_CONDA_CHANNELS) \
		--croot $(CONDA_BUILD_FOLDER) \
		--python $(PYTHON_VERSION) \
		--channel driver

release: test-in-clean-env git-tag build-release-and-upload

####################################################################################################
# test commands
#

test-in-clean-env: verify-conda-build-installed
	$(MAKE) setup-clean-env
	source activate test_env_$(PROJECT_NAME)_$(DEV_VERSION) && \
	$(MAKE) install-in-current-env && \
	$(MAKE) test-in-current-env && \
	source deactivate test_env_$(PROJECT_NAME)_$(DEV_VERSION)
	$(MAKE) teardown-clean-env

# run tests in the current environment
test-in-current-env:
	git lfs fetch
	pytest tests -vv

####################################################################################################
# helper commands
#

# make sure that we don't have uncommited changes, we'll want to check for un-added files in the future
verify-changes-commited:
	git diff-index --quiet HEAD --

verify-conda-build-installed:
	-source deactivate && conda install conda-build=$(CONDA_BUILD_VERSION) --yes
