"""Versioned frame contract for the rule engine and GPU locomotion service.

The six poses are consumed in order by the existing 120 Hz combat loop. The
whole reply must validate before any pose mutates the rule engine state.
"""
from dataclasses import dataclass
import json
import numpy as np

try:
    import orjson as _native_json
except ImportError:
    _native_json=None

FRAME_ENCODER='orjson-numpy' if _native_json is not None else 'json'

VERSION = 1
SUBSTEPS = 6
DT = 1 / 120


@dataclass(frozen=True)
class MotionFrame:
    ids: tuple
    positions: np.ndarray
    velocities: np.ndarray
    headings: np.ndarray
    folded: np.ndarray
    grounded: np.ndarray
    rotations: np.ndarray | None = None
    angular_velocities: np.ndarray | None = None
    legs: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self,'ids',tuple(self.ids))
        n = len(self.ids)
        if not n or len(set(self.ids)) != n or any(not isinstance(v, str) for v in self.ids):
            raise ValueError('Motion frame requires unique robot ids')
        for name, shape in [('positions',(SUBSTEPS,n,3)), ('velocities',(SUBSTEPS,n,3)),
                            ('headings',(SUBSTEPS,n)), ('folded',(SUBSTEPS,n))]:
            value = np.asarray(getattr(self,name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f'Invalid motion frame {name}')
            # Immutable byte ownership prevents a later writer from making
            # validated samples non-finite before native JSON encoding.
            value=np.frombuffer(value.tobytes(order='C'),dtype=np.float64).reshape(shape)
            if not np.isfinite(value).all():raise ValueError(f'Invalid motion frame {name}')
            object.__setattr__(self,name,value)
        if ((self.folded < 0) | (self.folded > 1)).any():
            raise ValueError('Fold progress must be in [0,1]')
        flags = np.asarray(self.grounded)
        if flags.shape != (SUBSTEPS,n) or flags.dtype != np.bool_:
            raise ValueError('Grounded must contain boolean samples')
        flags = np.frombuffer(flags.tobytes(order='C'),dtype=np.bool_).reshape(flags.shape)
        object.__setattr__(self,'grounded',flags)
        extra = (self.rotations, self.angular_velocities, self.legs)
        if any(v is not None for v in extra):
            if any(v is None for v in extra):
                raise ValueError('Version two requires complete attitude and leg samples')
            for name, width in (('rotations',4), ('angular_velocities',3), ('legs',2)):
                value = np.asarray(getattr(self,name), dtype=np.float64)
                if value.shape!=(SUBSTEPS,n,width) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid motion frame {name}')
                value = np.frombuffer(value.tobytes(order='C'),dtype=np.float64).reshape(value.shape)
                object.__setattr__(self,name,value)
            if not np.allclose(np.linalg.norm(self.rotations,axis=2),1.,atol=1e-4,rtol=0):
                raise ValueError('Motion rotation must be a unit quaternion')
            if (self.legs<-.01).any():
                raise ValueError('Negative leg length')

    def _header(self, sequence):
        if isinstance(sequence,bool) or not isinstance(sequence,int) or sequence < 0:
            raise ValueError('Invalid frame sequence')
        return dict(op='kinematic_frame_result',version=2 if self.rotations is not None else VERSION,sequence=sequence,dt=DT,ids=list(self.ids))

    def response(self, sequence):
        result = dict(self._header(sequence), positions=self.positions.tolist(),
                    velocities=self.velocities.tolist(), headings=self.headings.tolist(),
                    folded=self.folded.tolist(), grounded=self.grounded.tolist())
        if self.rotations is not None:
            result.update(rotations=self.rotations.tolist(), angular_velocities=self.angular_velocities.tolist(), legs=self.legs.tolist())
        return result

    def encoded_response(self, sequence):
        if _native_json is None:
            return json.dumps(self.response(sequence),allow_nan=False,separators=(',',':')).encode()+b'\n'
        payload=dict(self._header(sequence),positions=self.positions,velocities=self.velocities,
                     headings=self.headings,folded=self.folded,grounded=self.grounded)
        if self.rotations is not None:
            payload.update(rotations=self.rotations,angular_velocities=self.angular_velocities,legs=self.legs)
        return _native_json.dumps(payload,option=_native_json.OPT_SERIALIZE_NUMPY|_native_json.OPT_APPEND_NEWLINE)
