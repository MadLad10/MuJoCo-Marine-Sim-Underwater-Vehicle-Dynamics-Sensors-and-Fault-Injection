"""MuJoCo marine simulation: dynamics, noisy sensors, faults and annotations."""
__version__ = '0.1.0'
__all__ = ['MarineSim', 'Fault', 'FaultInjector', 'Referee']


def __getattr__(name):
    if name == 'MarineSim':
        from .sim import MarineSim
        return MarineSim
    if name in {'Fault', 'FaultInjector'}:
        from .faults import Fault, FaultInjector
        return {'Fault': Fault, 'FaultInjector': FaultInjector}[name]
    if name == 'Referee':
        from .referee import Referee
        return Referee
    raise AttributeError(name)
