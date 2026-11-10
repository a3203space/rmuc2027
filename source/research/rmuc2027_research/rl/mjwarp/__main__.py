"""Device smoke test only; this is not the Arena training environment."""
import argparse
import json
from .runtime import BatchPhysics


def main():
    parser = argparse.ArgumentParser(description='MJWarp physics device smoke test (not 5v5 training)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-worlds', type=int, default=8)
    args = parser.parse_args()
    xml = '<mujoco><worldbody><geom type="plane" size="10 10 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    try:
        with BatchPhysics(xml, args.num_worlds, args.device) as physics:
            for _ in range(20):
                physics.step(physics.np.zeros((args.num_worlds, 0)))
            state = physics.state()
            if not all(physics.np.isfinite(value).all() for value in state.values()):
                raise RuntimeError('Nonfinite physics state')
            print(json.dumps({'kind': 'physics-smoke-test', **physics.metadata()}))
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    main()
