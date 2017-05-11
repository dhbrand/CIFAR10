#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""

import sys
import base64

def main(argv):
    '''
    Encode a string to byte64
    '''
    for arg in argv:
        print(arg)
        out = base64.b64encode(bytes(arg, "utf-8"))
        print(out)

if __name__ == "__main__":
    main(sys.argv[1:])
