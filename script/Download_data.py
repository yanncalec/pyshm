#!/usr/bin/env python

import sys, os
from optparse import OptionParser       # command line arguments parser

__script__ = 'Download or update database from the server of OSMOS.'


def main():
    usage_msg = '{} [options] data_directory'.format(sys.argv[0])
    # example_msg = 'Example: '

    parser = OptionParser(usage_msg)

    parser.add_option('-p', '--PID', dest='PID', type='int', default=None, help='Project Key ID. If not given all projects presented in the destination database directory will be processed.')
    # parser.add_option('-a', '--assemble', dest='assemble', action='store_true', default=False, help='Assemble all pkl files of different Liris of the same PID into a single pkl file named \'Raw_latest.pkl\'.')
    parser.add_option('-f', '--force', dest='force', action='store_true', default=False, help='Force to assembling data of all sensors into a single file (even no new data are fetched).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('\nUsage: '+usage_msg)
        sys.exit(0)
    else:  # check datadir
        datadir = args[0]
        if not os.path.isdir(datadir):
            raise FileNotFoundError(datadir)

    # Lazy import to speed up the response of the script
    from pyshm.OSMOS_pkg.Download_data import Download_data

    Download_data(datadir, options=options)

if __name__ == '__main__':
    print(__script__)
    # print()
    main()
