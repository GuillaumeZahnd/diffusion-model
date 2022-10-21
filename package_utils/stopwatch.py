import math


def stopwatch(time_now, time_ini):
  time_delta        = time_now - time_ini
  time_minutes      = int(time_delta // 60)
  time_seconds      = int(time_delta % 60)
  time_milliseconds = int(1000*(time_delta - math.floor(time_delta)))
  return str(time_minutes).zfill(2) + ':' + str(time_seconds).zfill(2) + ':' + str(time_milliseconds).zfill(3)
