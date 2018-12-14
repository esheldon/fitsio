# Patches for cfitsio

This directory contains patches for the cfitsio build. These patches
are applied before the library is compiled during the python package
build step.

The patches were generated with the script `build_cfitsio_patches.py` by
Matthew Becker in December of 2018.

## Adding New Patches

To add new patches, you need to

1. Make a copy of the file you want to patch.
2. Modify it.
3. Call `diff -u old_file new_file` to a get a unified format patch.
4. Make sure the paths in the patch at the top look like this
    ```
    --- cfitsio<version>/<filename>	2018-03-01 10:28:51.000000000 -0600
    +++ cfitsio<version>/<filename>	2018-12-14 08:39:20.000000000 -0600
    ...
    ``` 
    where `<version>` and `<filename>` have the current cfitsio version and
    file that is being patched.

5. Commit the patch file in the patches directory with the name `<filename>.patch`.
