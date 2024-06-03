#!/bin/bash
set -e

PATCH_DIR=$(dirname $(readlink -f $0))

pip install openai-whisper==20231117
dist_packages=$(python3 -c "import site;print(site.getsitepackages()[0])")
dist_packages_path=$(echo "$dist_packages" | sed "s/[',\[\]]//g")
echo "dist-packages path is: $dist_packages_path"
if [ -d "$dist_packages_path" ]; then
    echo "do patch in: $dist_packages_path"
    cd $dist_packages_path
    patch -p0 < $PATCH_DIR/whisper.patch
else
    echo "dist-packages path is invalid"
fi