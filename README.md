# SketchNE
code for SketchNE

# Install

The code is compiled and run with g++ 7.4.0 (any supporting c++17 should work in theory).

## Install Boost
In the spectral propagation strategy, we need modified Bessel functions of the first kind which is supported by Boost.

```
sudo apt-get install libboost-dev
```

## Install Intel MKL
Intel MKL is used for basic linear algebra operations.
You can install with Anaconda

```
  conda create -n sketchne python=3.7 # first create a new python env

  conda activate sketchne # activate the new created env

  conda install mkl -c intel --no-update-deps

  conda install mkl-devel
```
You can also download directly from Intel. Please follow
```
https://software.intel.com/en-us/mkl/choose-download/linux
```
The installation script will install intel mkl (by default) at `/opt/intel`.

# Compile

To compile sketchne, you may need to edit Makefile when you install MKL with Anaconda. You need to set something like:
```
INCLUDE_DIRS = -I./ligra -I./pbbslib -I./mklfreigs -I"{ANACONDA_PATH}/envs/sketchne/include"
LINK_DIRS = -L"{ANACONDA_PATH}/envs/sketchne/lib"
```
Then run `make` to compile.

To clean the compiled file, run `make clean`.

Before running the example, you may need to set environment. If you install MKL directly from Intel, you can set:
```
export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64
```
Or you can set the library path in anaconda path:
```
export LD_LIBRARY_PATH={ANACONDA_PATH}/envs/sketchne/lib
```

# Run

## Example
run `blog.sh` in example directory.

The input format is the adjacency graph format used by [GBBS](https://github.com/ParAlg/gbbs). All vertices and offsets are 0 based and represented in decimal. The specific format is as follows:

```
AdjacencyGraph
<n>
<m>
<o0>
<o1>
...
<o(n-1)>
<e0>
<e1>
...
<e(m-1)>
```
We have a format conversion program in the util directory, which supports the conversion of edgelist and mat formats to adjacency graph format.

Here we only give the small graph as an example. If you need more datasets for testing, please download and unzip datasets used in NetSMF paper.
```
cd data_bin
wget https://sampledbsql1backup.blob.core.windows.net/www19netsmf/datasets.zip
unzip datasets.zip
```
It's easy to found `youtube.mat (youtube dataset)` and `mag.edge (OAG dataset)` in the datasets.

`friendster` and `livejournal` can be download from SNAP: `https://snap.stanford.edu/data/`.


## Graphs with more than 10 billion edges
ClueWeb graph can be downloaded from [here](http://law.di.unimi.it/webdata/clueweb12/).

Hyperlink2014 graph can be downloaded from [here](http://webdatacommons.org/hyperlinkgraph/2014-04/download.html).

Hyperlink2012 graph can be downloaded from [here](http://webdatacommons.org/hyperlinkgraph/2012-08/download.html).

