#!/bin/bash

# Get folder where the install script is located
dname=$(realpath $(dirname $0))
cd $HOME
rm -rf .hdf5plot
mkdir .hdf5plot
cd .hdf5plot
if command -v uv &> /dev/null; then
    echo "uv found, using it instead of pip."
    USE_UV=1
    uv venv hdf5plotvenv
else
    USE_UV=0
    python3 -m venv hdf5plotvenv
fi
. hdf5plotvenv/bin/activate
if [[ "$(which python3)" != "$HOME/.hdf5plot/hdf5plotvenv/bin/python3" ]]; then
    echo "Failed to enter venv."
    exit 1
fi

echo "Created venv at $(which python3)"
if [[ "$USE_UV" == "1" ]]; then
    uv pip install --upgrade pip setuptools wheel
    uv pip install -r "${dname}/requirements.txt"
    uv pip install "${dname}"
else
    pip install --upgrade pip
    pip install --upgrade setuptools
    pip install --upgrade wheel

    pip install -r "${dname}/requirements.txt"
    pip install "${dname}"
fi
mkdir -p $HOME/.local/bin
cp "${dname}/hdf5plot" $HOME/.local/bin
# cp "${dname}/hdf5_plotter.py" $HOME/.hdf5plot/
mkdir -p $HOME/.local/share/applications
cp "${dname}/crizz-hdf5plotter.desktop" $HOME/.local/share/applications