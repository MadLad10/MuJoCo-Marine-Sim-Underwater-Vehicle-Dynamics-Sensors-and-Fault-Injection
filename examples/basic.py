"""Run from an installed package: python examples/basic.py."""
import os
os.environ.setdefault('MUJOCO_GL', 'disable')

from auv_marine_sim import MarineSim


def main():
    sim = MarineSim('default', seed=0, faults=[
        {'type': 'dvl_dropout', 't_start': .5},
    ])
    try:
        sim.run_for(1.)
        print('DVL valid:', sim.readings['dvl']['valid'])
        print('Referee:', sim.referee.report())
    finally:
        sim.close()


if __name__ == '__main__':
    main()
