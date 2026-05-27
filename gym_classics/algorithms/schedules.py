import numpy as np

class Schedule:
    """Base class for schedules."""
    def __call__(self, t):
        """Returns the scheduled value at time step (or episode) t."""
        raise NotImplementedError

class ConstantSchedule(Schedule):
    """A schedule that always returns a constant value."""
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, t):
        return self.value

class StepSchedule(Schedule):
    """A schedule that always returns a constant value."""
    def __init__(self, high_value, low_value, steps):
        self.high_value = float(high_value)
        self.low_value = float(low_value)
        self.steps = int(steps)

    def __call__(self, t):
        if t < self.steps:
            return self.high_value
        return self.low_value

class LinearDecaySchedule(Schedule):
    """A schedule that decays linearly from initial_value to min_value over decay_steps."""
    def __init__(self, initial_value, min_value, decay_steps):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)
        self.decay_steps = int(decay_steps)

    def __call__(self, t):
        fraction = min(float(t) / max(1, self.decay_steps), 1.0)
        return self.initial_value + fraction * (self.min_value - self.initial_value)

class ExponentialDecaySchedule(Schedule):
    """A schedule that decays exponentially with a given decay_rate.
    Value at time t is max(min_value, initial_value * (decay_rate ** t)).
    """
    def __init__(self, initial_value, min_value, decay_rate):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)
        self.decay_rate = float(decay_rate)

    def __call__(self, t):
        return max(self.min_value, self.initial_value * (self.decay_rate ** t))

class InverseDecaySchedule(Schedule):
    """A schedule that decays inversely proportional to t.
    Value at time t is max(min_value, initial_value / t) for t > 0.
    """
    def __init__(self, initial_value, min_value=0.0):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)

    def __call__(self, t):
        if t == 0:
            return self.initial_value
        return max(self.min_value, self.initial_value / t)

def plot_schedule(schedule, steps=1000):
    """Plots the values of a schedule over a given number of steps.
    
    Example:
        from gym_classics.algorithms.schedules import ConstantSchedule, LinearDecaySchedule, ExponentialDecaySchedule, InverseDecaySchedule, plot_schedule
        
        # Create different schedules
        constant_sched = ConstantSchedule(0.1)
        linear_sched = LinearDecaySchedule(initial_value=1.0, min_value=0.1, decay_steps=1000)
        exponential_sched = ExponentialDecaySchedule(initial_value=1.0, min_value=0.1, decay_rate=0.99)
        inverse_sched = InverseDecaySchedule(initial_value=10.0, min_value=0.1)
        
        # Plot them
        plot_schedule(constant_sched, steps=500)
        plot_schedule(linear_sched, steps=1000)
        plot_schedule(exponential_sched, steps=1000)
        plot_schedule(inverse_sched, steps=1000)

    Args:
        schedule: A Schedule instance (or any callable taking an integer step).
        steps: The number of steps to plot.
    """
    import matplotlib.pyplot as plt
    
    values = [schedule(t) for t in range(steps)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(steps), values, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Schedule Value')
    plt.title('Schedule Plot')
    plt.grid(True)
    plt.show()