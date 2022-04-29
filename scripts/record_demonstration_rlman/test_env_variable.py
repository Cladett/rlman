"""
@brief    Script to test how to set and read envirnoment variables. 
          It sets an env. variable and launches a terminal.
@author   Claudia D'Ettorre (c.dettorre@ucl.ac.uk).
@date     1 Sep 2020.
"""

import os
import pty
import sys

def main():
    sys.stdout.write('prova')
    sys.stderr.write('prova1')
    sys.stdout.flush()
    sys.stderr.flush()
    #os.environ['CACA'] = 'foo'
    #pty.spawn('/bin/zsh')

if __name__ == '__main__':
    main() 
