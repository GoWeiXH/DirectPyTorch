
CANDIDATE = "[acc, val_acc, loss, val_loss]"


class MonitorError(Exception):

    def __init__(self, value):
        super(MonitorError, self).__init__()
        self.value = value

    def __str__(self):
        return "Monitor '{}' is invalid, Monitor must be selected from {}".format(
            self.value, CANDIDATE)


class MetricsError(Exception):

    def __init__(self, value):
        super(MetricsError, self).__init__()
        self.value = value

    def __str__(self):
        return "Metrics '{}' is invalid, Metrics must be selected from {}".format(
            self.value, CANDIDATE)


class NoCompileError(Exception):

    def __init__(self):
        super(NoCompileError, self).__init__()

    def __str__(self):
        return "Model must be compiled before training, 'Model().compile(**args)'"


class ActivationTypeError(Exception):

    def __init__(self):
        super(ActivationTypeError, self).__init__()

    def __str__(self):
        return "Activation function is invalid, should be str or torch.nn.Module"


class ActivationError(Exception):

    def __init__(self, value):
        super(ActivationError, self).__init__()
        self.value = value

    def __str__(self):
        return "Activation '{}' is is not supported, you can try to use the PyTorch's activation function.\n" \
               "Such as use torch.nn.Sigmoid() instead of 'sigmoid' ".format(self.value)
