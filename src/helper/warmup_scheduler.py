import math


class WarmupCosineAnnealingScheduler:
    def __init__(self, optimizer, max_steps, warmup, learning_rate):
        self.optimizer = optimizer
        self._step = 0
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.warmup = warmup
        self._rate = 0

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def rate(self, step=None):
        if step is None:
            step = self._step

        if step < self.warmup:
            return self.learning_rate * step / self.warmup
        if step > self.max_steps:
            return self.learning_rate / 10

        decay_ratio = (step - self.warmup) / (self.max_steps - self.warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.learning_rate / 10 + coeff * (
            self.learning_rate - self.learning_rate / 10
        )
