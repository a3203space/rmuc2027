"""Versioned frame contract for the rule engine and GPU locomotion service.

The six poses are consumed in order by the existing 120 Hz combat loop. The
whole reply must validate before any pose mutates the rule engine state.
"""
from dataclasses import dataclass
import numpy as np

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

    def __post_init__(self):
        n = len(self.ids)
        if not n or len(set(self.ids)) != n or any(not isinstance(v, str) for v in self.ids):
            raise ValueError('Motion frame requires unique robot ids')
        for name, shape in [('positions',(SUBSTEPS,n,3)), ('velocities',(SUBSTEPS,n,3)),
                            ('headings',(SUBSTEPS,n)), ('folded',(SUBSTEPS,n))]:
            value = np.array(getattr(self,name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid motion frame {name}')
            value.setflags(write=False)
            object.__setattr__(self,name,value)
        if ((self.folded < 0) | (self.folded > 1)).any():
            raise ValueError('Fold progress must be in [0,1]')
        flags = np.asarray(self.grounded)
        if flags.shape != (SUBSTEPS,n) or flags.dtype != np.bool_:
            raise ValueError('Grounded must contain boolean samples')
        flags = flags.copy(); flags.setflags(write=False)
        object.__setattr__(self,'grounded',flags)

    def response(self, sequence):
        if isinstance(sequence,bool) or not isinstance(sequence,int) or sequence < 0:
            raise ValueError('Invalid frame sequence')
        return dict(op='kinematic_frame_result', version=VERSION, sequence=sequence,
                    dt=DT, ids=list(self.ids), positions=self.positions.tolist(),
                    velocities=self.velocities.tolist(), headings=self.headings.tolist(),
                    folded=self.folded.tolist(), grounded=self.grounded.tolist())
