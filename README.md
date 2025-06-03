# RACE
This repository contains three python scripts I used to perform image reconstructions of motion blurred X-ray projections. 
Two different reconstruction methods are implemented, namely ARTIC [1] and RACE [2].

The script "continuous_rotation.py' first simulates X-ray projections of a certain phantom, using the astra toolobox [3]. 
The simulated projections are then blurred by averaging the intensities, simulating object motion (only rotational) during the exposure time.
The blurred projections are then reconstructed, without taking the object motion into account, resulting in a blurred reconstruction.
Then ARTIC is used to reconstruct the same set of blurred projections, resulting the removal of a large part of the motion artefacts.
Finally, RACE is used and results in an even better reconstruction, especially for a lower number of projections and consequently a higher amuont of blur.

The script "continuous_combined motion.py" essentially does the same, however this time the motion of the object is not only rotational, but also translational.
During the simulation the object shifts from left to right parallel to the detector. The folder "Phantoms" contains different phantoms to test the code with.

The script "flexct_continuous_rottrans_ndeg.py" is designed to perform an ARTIC and RACE reconstruction of the continuously acquired projections of the data set at: https://zenodo.org/records/12918504.
This data was measured with the FleXCT scanner [4] and comprises of a reference set of projections, that can be reconstructed into a reference image with negligible blurring, and a set of blurred projections.
These blurred projections can be averaged together to increase the blurring even more.
Most of the settings needed for the reconstruction are automatically read from the acquisition file. Although, for the blurred dataset, this did not work properly and the settings needed to be specified manually.
The projection geometries for 1, 2 and 4 averages are available in the folder "FleXCT_geometries".
This folder contains the projection geometry to do the blurred reconstruction, the set of projection geometries (that need to be averaged) for the ARTIC reconstruction and the subsampled projection geometries for the RACE reconstruction.

Additionally, the folder "ARTIC_original" contains the orignal MATLAB code used for ARTIC in [1].

Finally, the script "synchrotron_RACE.py" contains the code to apply ARTIC and RACE on data measured with a parallel beam geometry, such as synchrotron data.
Here, the center of rotation (COR) is not yet fully implemented in a proper way (todo).

To run the code, the imsolve package is required, which can be found at: https://github.com/RendersJens/imsolve/tree/main

Please cite [2], when using or adapting this code in  your own research.


# REFERENCES
[1] J. Cant, et al., "Modeling blurring effects due to continuous gantry rotation: Application to region of interest tomography", Med. Phys. 42, 2709-2717 (2015)

[2] B Huyge, et al., "X-ray image reconstruction for continuous acquisitions with a generalized motion model", Opt. Express, (Under review) (2024)

[3] W. van Aarle, et al., “Fast and flexible X-ray tomography using the ASTRA toolbox”, Opt. Express 24, 25129–25147 (2016)

[4] B. De Samber, et al., “FleXCT: a flexible X-ray CT scanner with 10 degrees of freedom”, Opt. Express 29, 3438–3457 (2021).
