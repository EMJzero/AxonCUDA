#!/bin/bash

wget -O snns.tar.gz https://zenodo.org/records/19194881/files/snns.tar.gz?download=1
echo "Un-tarring SNNs, might take a while, sit back and relax *.* !"
tar -xjf snns.tar.gz
mv processed_snns snns
rm snns.tar.gz

mkdir ispd98_16x
mkdir ispd98 && cd ispd98
wget -O ispd98.tar.gz https://zenodo.org/records/30176/files/ISPD98_hypergraphs.tar.gz?download=1
echo "Un-tarring hypergraphs, might take a while, sit back and relax *.* !"
tar -xzf ispd98.tar.gz
echo "Scaling up hypergraphs, might also take a while, sit back and relax °.° !"
for f in *; do
    [ -f "$f" ] || continue
    python3 ../scale_hgr.py "$f" "../ispd98_16x/$f" 16
done
cd ..