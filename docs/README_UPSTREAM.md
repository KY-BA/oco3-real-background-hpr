# Inversion of CO2 plumes - OCO3-SAM and synthetic data

Official repository for the paper "Quantification of CO2 hotspot emissions from OCO-3 SAM CO2 satellite images using deep learning methods" submitted to "Geoscientific Model Development".
 
This project presents the development and application of a deep learning-based method for inverting CO2 atmospheric plumes
from power plants using satellite imagery of the CO2 total column mixing ratios (XCO2).

Our scripts and modules are written in Python, using Tensorflow as the deep learning framework.

To employ these scripts and train CNN models, download the datasets of fields and plumes from [inv-zenodo](https://doi.org/10.5281/zenodo.12788520).
For direct testing, weights of trained models can also be obtained from [inv-zenodo](https://doi.org/10.5281/zenodo.12788520).

After data collection/generation, 'main.py` is used to train the CNN (config constructed with hydra).

For any queries, do not hesitate to reach out.

## Acknowledgements and Authors

This project is funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement 958927 (Prototype system for a Copernicus CO2 service). 
CEREA is a proud member of Institut Pierre Simon Laplace (IPSL).

## Support

Feel free to contact: joffrey.dumont@ecmwf.int