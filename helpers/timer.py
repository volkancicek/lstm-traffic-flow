import datetime as dt


class Timer:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = dt.datetime.now()

    def stop(self):
        end_time = dt.datetime.now()
        print('duration : %s' % (end_time - self.start_time))
