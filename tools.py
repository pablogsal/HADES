__author__ = 'pablogsal'

import sys,time,random
import logging
import numpy as np

typing_speed = 1200 #wpm

def stype(t):

    if not qflag :
        print(t)
    else:
        pass

# def stype(t):
#     for l in t:
#         sys.stdout.write(l)
#         sys.stdout.flush()
#         time.sleep(random.random()*10.0/typing_speed)
#     #sys.stdout.write('.....Done.')
#     sys.stdout.write('\n')



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


def construct_lines_from_angle(angle,background=np.array(0),step=6,scale=5):

    # Get the angle array dimensions to go over it
    (dim_y,dim_x)=angle.shape

    #Initialize the list of lines

    lines=[]

    #Main loop over angle array
    for y in range(0,dim_y,step):
        for x in range(0,dim_x,step):

         #If there is a limitator get it, else initialize the limitator to 1
         # Here the limitator is a boolean with 0 where we do not want to get lines

            if background.shape != ():

                if background[y,x] == 0:
                    boole_limitator=0
                else:
                    boole_limitator=1
            else:
                boole_limitator=1



            # We must eliminate the border lines to avoid problems with scaling the plot

            if x <1 or x>dim_x-2 or y<1 or y > dim_y-2 :
                boole_limitator =0

            #Get the line lenght
            line_lenght=1*0.5*scale*boole_limitator

            #Get the angle
            theta=angle[y,x]

            #Construct the segment
            x1=x+line_lenght*np.sin(theta)
            y1=y-line_lenght*np.cos(theta)
            x2=x-line_lenght*np.sin(theta)
            y2=y+line_lenght*np.cos(theta)
            line=[(x2,y2),(x1,y1)]

            # Add the segment to the liner accumulator
            lines.append(line)

    return lines
