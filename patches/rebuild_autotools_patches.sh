#!/usr/bin/env bash

cfitsio_ver="4.6.4"
cfitsio_dir=cfitsio-${cfitsio_ver}

rm -rf ${cfitsio_dir}
rm -rf ${cfitsio_dir}-build
tar -xvf ${cfitsio_dir}.tar.gz

cp -r ${cfitsio_dir} ${cfitsio_dir}-build

for fname in configure.ac Makefile.am; do
    patch ${cfitsio_dir}-build/${fname} patches/${fname}.patch
done

pushd ${cfitsio_dir}-build

autoreconf -i

popd

for fname in configure Makefile.in; do
    diff -u ${cfitsio_dir}-build/${fname} ${cfitsio_dir}/${fname} > patches/${fname}.patch
done

rm -rf ${cfitsio_dir}
rm -rf ${cfitsio_dir}-build
