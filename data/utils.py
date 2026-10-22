# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import os
import sys
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
import xarray as xr
from icecream import ic

from include.loss import calculate_weighted_plume


def get_clouds_array(
    dir_clouds_array: str,
    dataset_split: str,
    clouds_threshold: Union[float, List[float]],
    size_image: int,
):
    """Return clouds array at right dataset split and clouds threshold."""

    if type(clouds_threshold) == float:
        formatted_threshold = f"{int(clouds_threshold * 100):03d}"
        path_clouds_array = os.path.join(
            dir_clouds_array,
            f"extracted_clouds_{dataset_split}_s{size_image}_t{formatted_threshold}.npy",
        )
    elif type(clouds_threshold) == list:
        formatted_threshold_min = f"{int(clouds_threshold[0] * 100):03d}"
        formatted_threshold_max = f"{int(clouds_threshold[1] * 100):03d}"
        path_clouds_array = os.path.join(
            dir_clouds_array,
            f"extracted_clouds_{dataset_split}_s{size_image}_tmin{formatted_threshold_min}_tmax{formatted_threshold_max}.npy",
        )
        ic(path_clouds_array)
    else:
        print(clouds_threshold)
        ic(type(clouds_threshold))
        logging.info("Error to get clouds array due to clouds threshold input.")
        sys.exit()
    clouds_array = np.load(path_clouds_array)
    return clouds_array


def get_xco2_noisy(ds: xr.Dataset):
    """Return noisy xco2 field related to ds."""
    xco2 = np.expand_dims(ds.xco2_noisy.values, -1)
    return xco2


def get_xco2_noisy_prec(ds: xr.Dataset):
    """Return noisy xco2 field related to ds at the previous time."""
    xco2 = np.concatenate((ds.xco2_noisy.values[0:1], ds.xco2_noisy.values[0:-1]))
    xco2 = np.expand_dims(xco2, -1)
    return xco2


def get_xco2_noiseless(ds: xr.Dataset) -> np.ndarray:
    """Return noiseless xco2 field related to ds."""
    return np.expand_dims(ds.xco2.values, -1)


def get_xco2_noiseless_prec(ds: xr.Dataset):
    """Return noiseless xco2 field related to ds at the previous time."""
    xco2 = np.concatenate((ds.xco2.values[0:1], ds.xco2.values[0:-1]))
    return np.expand_dims(xco2, -1)


def get_no2_noisy(ds: xr.Dataset) -> np.ndarray:
    """Return noisy no2 field related to ds."""
    no2 = np.expand_dims(ds.no2_noisy.values, -1)
    return no2


def get_no2_noisy_prec(ds: xr.Dataset) -> np.ndarray:
    """Return noisy no2 field related to ds."""
    no2 = np.concatenate((ds.no2_noisy.values[0:1], ds.no2_noisy.values[0:-1]))
    return np.expand_dims(no2, -1)


def get_seg_pred_no2(ds: xr.Dataset) -> np.ndarray:
    """Return no2 model segmentations from ds."""
    return np.expand_dims(ds.seg_pred_no2.values, -1)


def get_seg_pred_no2_prec(ds: xr.Dataset) -> np.ndarray:
    """Return no2 model segmentations from ds."""
    seg_pred_no2 = np.concatenate(
        (ds.seg_pred_no2.values[0:1], ds.seg_pred_no2.values[0:-1])
    )
    return np.expand_dims(seg_pred_no2, -1)


def get_no2_noiseless(ds: xr.Dataset) -> np.ndarray:
    """Return noiseless no2 field related to ds."""
    return np.expand_dims(ds.no2.values, -1)


def get_u_wind(ds: xr.Dataset) -> np.ndarray:
    """Return u wind related to ds."""
    return np.expand_dims(ds.u.values, -1)


def get_u_wind_prec(ds: xr.Dataset) -> np.ndarray:
    """Return u wind related to ds."""
    u = np.concatenate((ds.u.values[0:1], ds.u.values[0:-1]))
    return np.expand_dims(u, -1)


def get_v_wind(ds: xr.Dataset) -> np.ndarray:
    """Return u wind related to ds."""
    return np.expand_dims(ds.v.values, -1)


def get_v_wind_prec(ds: xr.Dataset) -> np.ndarray:
    """Return v wind related to ds."""
    v = np.concatenate((ds.v.values[0:1], ds.v.values[0:-1]))
    return np.expand_dims(v, -1)


def get_xco2_plume(ds: xr.Dataset) -> np.ndarray:
    """Return plume from ds."""
    return np.expand_dims(ds.plume.values, -1)


def get_no2_plume(ds: xr.Dataset) -> np.ndarray:
    """Return noisy no2 plume related to ds."""
    no2 = np.expand_dims(ds.no2_plume.values, -1)
    return no2


def get_plume_prec(ds: xr.Dataset) -> np.ndarray:
    """Return plume from ds at the previous time."""
    plume = np.concatenate((ds.plume.values[0:1], ds.plume.values[0:-1]))
    return np.expand_dims(plume, -1)


def get_xco2_back(ds: xr.Dataset) -> np.ndarray:
    """Return xco2 back from ds."""
    return np.expand_dims(ds.xco2_back.values, -1)


def get_xco2_back_prec(ds: xr.Dataset) -> np.ndarray:
    """Return xco2_back from ds at the previous time."""
    xco2_back = np.concatenate((ds.xco2_back.values[0:1], ds.xco2_back.values[0:-1]))
    return np.expand_dims(xco2_back, -1)


def get_xco2_alt_anthro(ds: xr.Dataset) -> np.ndarray:
    """Return xco2_alt_anthro back from ds."""
    return np.expand_dims(ds.xco2_alt_anthro.values, -1)


def get_xco2_alt_anthro_prec(ds: xr.Dataset) -> np.ndarray:
    """Return xco2_alt_anthro from ds at the previous time."""
    xco2_alt_anthro = np.concatenate(
        (ds.xco2_alt_anthro.values[0:1], ds.xco2_alt_anthro.values[0:-1])
    )
    return np.expand_dims(xco2_alt_anthro, -1)


def get_bool_perf_seg(ds: xr.Dataset) -> np.ndarray:
    """Return boolean perfect segmentations from ds."""
    return np.expand_dims(ds.bool_perf_seg.values, -1)


def get_emiss(ds: xr.Dataset, N_hours_prec: int) -> np.ndarray:
    """Return emiss array related to ds."""
    emiss = np.array(ds.emiss.values, dtype=float)
    emiss = emiss[:, 1 : N_hours_prec + 1]
    return emiss


def get_bool_(ds: xr.Dataset, N_hours_prec: int) -> np.ndarray:
    """Return emiss array related to ds."""
    emiss = np.array(ds.emiss.values, dtype=float)
    emiss = emiss[:, 1 : N_hours_prec + 1]
    return emiss


def get_weighted_plume(
    ds: xr.Dataset,
    curve: str = "linear",
    min_w: float = 0,
    max_w: float = 1,
    param_curve: float = 1,
):
    """Get modified plume matrices label output."""
    y_data = calculate_weighted_plume(
        np.array(ds.plume.values, dtype=float), min_w, max_w, curve, param_curve
    )
    return y_data


def get_weighted_plume_prec(
    ds: xr.Dataset,
    curve: str = "linear",
    min_w: float = 0,
    max_w: float = 1,
    param_curve: float = 1,
):
    """Get modified plume matrices label output."""
    plume = np.concatenate((ds.plume.values[0:1], ds.plume.values[0:-1]))
    y_data = calculate_weighted_plume(
        np.array(plume, dtype=float), min_w, max_w, curve, param_curve
    )
    return y_data


def get_timedate_vector(ds: xr.Dataset):
    """Get timedate vector in an xarray dataset"""

    hours_in_day = 24
    days_in_week = 7
    days_in_year = 365

    hour_angle = 2 * np.pi * ds["time"].dt.hour.values / hours_in_day
    weekday_angle = 2 * np.pi * ds["time"].dt.weekday.values / days_in_week
    yearday_angle = 2 * np.pi * ds["time"].dt.dayofyear.values / days_in_year

    timedate_vector = np.array(
        [
            np.cos(hour_angle),
            np.sin(hour_angle),
            np.cos(weekday_angle),
            np.sin(weekday_angle),
            np.cos(yearday_angle),
            np.sin(yearday_angle),
        ]
    )

    return timedate_vector


def get_inversion_model(
    dir_res: str,
    name_w: str = "w_last.h5",
    optimiser: str = "adam",
    loss=tf.keras.losses.MeanAbsoluteError(),
):
    """Get inversion neural network model."""
    model = tf.keras.models.load_model(
        os.path.join(dir_res, name_w),
        compile=False,
    )
    assert isinstance(model, tf.keras.models.Model)
    model.compile(optimiser, loss=loss)
    return model
