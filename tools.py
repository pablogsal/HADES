__author__ = 'pablogsal'

import sys,time,random
import logging


typing_speed = 1200 #wpm
def stype(t):
    for l in t:
        sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(random.random()*10.0/typing_speed)
    #sys.stdout.write('.....Done.')
    sys.stdout.write('\n')



#Class for output colors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Custom logger classes for error and OK

logging.addLevelName(60, "ERROR")
def log_error(self, message, *args, **kws):
    if self.isEnabledFor(60):
        self._log(60, bcolors.FAIL+message+bcolors.ENDC, args, **kws)

logging.addLevelName(25, "CHECK")
def log_OK(self, message, *args, **kws):
    if self.isEnabledFor(25):
        self._log(25, bcolors.OKGREEN+message+bcolors.ENDC, args, **kws)

logging.addLevelName(35, "WARNING")
def log_WARNING(self, message, *args, **kws):
    if self.isEnabledFor(35):
        self._log(35, bcolors.WARNING+message+bcolors.ENDC, args, **kws)



def bytes2human(n, format="%(value)i%(symbol)s"):
    """
    >>> bytes2human(10000)
    '9K'
    >>> bytes2human(100001221)
    '95M'
    """
    symbols = (' Bytes', ' KBytes', ' MBytes', ' GBytes', ' TBytes', ' PBytes', ' EBytes', ' ZBytes', ' YBytes')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def exit():
    sys.exit("We have found some errors. \nTry looking at the previous lines to find some clue. :("+'\n')