import fitsio
import numpy as np

seed = 42
rng = np.random.RandomState(seed=seed)

data = rng.normal(size=(100, 100))

fitsio.write(
    f"test_rice_dither2_seed{seed}.fits.fz",
    data,
    compress="RICE",
    qmethod="SUBTRACTIVE_DITHER_2",
    dither_seed=seed,
    clobber=True,
)

cdata = fitsio.read(f"test_rice_dither2_seed{seed}.fits.fz")

fitsio.write(
    f"test_rice_dither2_seed{seed}.fits",
    cdata,
    clobber=True,
)
