"""Lossless numeric EXR depth and strict categorical decoding."""

from pathlib import Path

import numpy as np

from ..validation import ContractError, require


def read_exr_channel(path: str|Path, channel: str='R') -> np.ndarray:
    import OpenEXR
    try:
        with OpenEXR.File(str(path),separate_channels=True) as file:
            channels=file.channels()
            require(channel in channels,str(path),f'missing EXR channel {channel}')
            return np.array(channels[channel].pixels,dtype=np.float32,copy=True)
    except (OSError,RuntimeError) as error:
        raise ContractError(str(path),f'cannot read EXR: {error}') from None


def write_depth_exr(path: str|Path, depth: np.ndarray) -> None:
    import OpenEXR
    depth=np.ascontiguousarray(depth,dtype=np.float32)
    require(depth.ndim==2 and bool(np.isfinite(depth).all()),'depth','expected finite 2D camera Z')
    with OpenEXR.File({'compression':OpenEXR.ZIP_COMPRESSION,'type':OpenEXR.scanlineimage,
                       'depth_convention':'camera_z_m'}, {'Z':depth}) as file:
        file.write(str(path))


def decode_passes(raw_depth_cm: np.ndarray, raw_ids: np.ndarray, codes: set[int], *, near_m: float, far_m: float):
    require(raw_ids.ndim==2 and raw_ids.shape==raw_depth_cm.shape,'capture','depth/mask resolution mismatch')
    require(bool(np.isfinite(raw_ids).all()),'lithology','nonfinite ID output')
    rounded=np.rint(raw_ids)
    require(bool((np.abs(raw_ids-rounded)<=0.005).all()),'lithology','noninteger ID output; exposure/tonemapping/blending may be enabled')
    require(set(np.unique(rounded)) <= {0,*codes},'lithology','unknown class in ID output')
    mask=rounded.astype(np.uint16)
    foreground=mask>0
    depth=np.asarray(raw_depth_cm,dtype=np.float32)/100.
    # Float32 depth projection and cm -> m conversion can straddle an exact clip
    # boundary by a few ULPs. Retain the measurement; only relax validation.
    near_error=4*np.finfo(np.float32).eps*max(1.,near_m)
    far_error=4*np.finfo(np.float32).eps*max(1.,far_m)
    require(bool((np.isfinite(depth[foreground]) & (depth[foreground]>=near_m-near_error) & (depth[foreground]<=far_m+far_error)).all()),
            'depth','foreground has invalid/out-of-clip camera Z')
    depth=np.where(foreground,depth,0).astype(np.float32)
    return depth,foreground.astype(np.uint8)*255,mask
