#!/usr/bin/env python

import sys, os
from optparse import OptionParser       # command line arguments parser

__script__ = 'Decomposition of static data into seasonal and trend components.'

def main():
    usage_msg = '{} [options] directory_of_PID'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    parser.add_option('--mwsize', dest='mwsize', type='int', default=24, help='Size of the moving window (default=24).')
    parser.add_option('--kzord', dest='kzord', type='int', default=1, help='Order of the recursive KZ filter (default=1).')
    parser.add_option('--lambda', dest='lamb', type='int', default=129600*30, help='Penality constant for the HP filter (default=129600*30).')
    parser.add_option('--causal', dest='causal', action='store_true', default=False, help='Use causal moving average.')
    parser.add_option('--method', dest='method', type='string', default='moving_average', help='Method of decomposition: moving_average (default), moving_median, hp_filter.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('\nUsage: '+usage_msg)
        sys.exit(0)
    else:  # check datadir
        datadir = args[0]
        if not os.path.isdir(datadir):
            raise FileNotFoundError(datadir)

    from pyshm.OSMOS_pkg.Decomposition_of_static_data import Decomposition_of_static_data
    Decomposition_of_static_data(datadir, options=options)


if __name__ == '__main__':
    print(__script__)
    print()
    main()
