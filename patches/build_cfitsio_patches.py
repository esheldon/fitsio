import os
import sys
import subprocess

VERSION = '3.47'
SRC_URL = (
    "https://heasarc.gsfc.nasa.gov/FTP/software/"
    "fitsio/c/cfitsio-%s.tar.gz" % VERSION)
SRC_TARBALL = os.path.basename(SRC_URL)
SRC_DIR = os.path.basename(SRC_URL).replace('.tar.gz', '')

# download
os.system(
    'rm -rf %s && rm -f %s && wget %s && tar xzvf %s && ls -alh' % (
        SRC_DIR, SRC_TARBALL, SRC_URL, SRC_TARBALL))

# diff src files
# the sources are all at the top level
os.makedirs('patches', exist_ok=True)

for root, _, files in os.walk(SRC_DIR):
    print(files)
    for fname in files:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join('cfitsio-%spatch' % VERSION, fname)
        patch = os.path.join('patches', fname + '.patch')
        os.system('diff -u %s %s > %s' % (src, dst, patch))
        with open(patch, 'rb') as fp:
            buff = fp.read()
        if len(buff) == 0:
            os.remove(patch)
    break

# clean up
os.system('rm -rf %s && rm -f %s' % (SRC_DIR, SRC_TARBALL))
