
class MonitorError(Exception):

    def __init__(self, value):
        super(MonitorError, self).__init__()
        self.value = value
        self.candidate = "[acc, val_acc, loss, val_loss]"

    def __str__(self):

        return "Monitor '{}' is invalid, Monitor must be selected from {}".format(
            self.value, self.candidate)
