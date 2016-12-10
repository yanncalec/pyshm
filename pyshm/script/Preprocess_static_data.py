#!/usr/bin/env python

"""Apply preprocessing and assemble processed data.
"""

# class Options:
#     sflag=False
#     oflag=False
#     tflag=False
#     jflag=False
#     verbose=False


def Preprocess_static_data(projdir, options):
    """Apply preprocessing and assemble processed data.

    Args:
        projdir (string): the directory of a project in the database
    Outputs:
        A file named Processed.pkl in projdir.
    """

    import glob, pickle
    import colorama
    import copy

    from pyshm import OSMOS

    Sdata = {}

    # Load assembled raw data
    Rdata, Sdata_raw, Ddata, Locations = OSMOS.load_raw_data(os.path.join(projdir, 'Raw.pkl'))

    for loc, X in Sdata_raw.items():
        if options.verbose:
            print('Processing location {}...'.format(loc))
        if len(X)>0:
            try:
                Sdata[loc] = OSMOS.static_data_preprocessing(X,
                                                            sflag=options.sflag,
                                                            oflag=options.oflag,
                                                            jflag=options.jflag,
                                                            tflag=options.tflag,
                                                            nh=options.nh)

            except Exception as msg:
                print(colorama.Fore.RED + 'Error: ', msg)
                print(colorama.Style.RESET_ALL)
                # raise Exception

    fname = os.path.join(projdir,'Preprocessed_static.pkl')
    # with open(fname, 'wb') as fp:
    #     pickle.dump({'Data': Sdata}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fname, 'wb') as fp:
        pickle.dump({'Data': Sdata}, fp)

    if options.verbose:
        print('Results saved in {}'.format(fname))


__all__ =["Prepreocess_static_data"]

__script__ = __doc__

import sys, os
from optparse import OptionParser       # command line arguments parser


def main():
    usage_msg = '{} [options] <directory_of_PID>'.format(sys.argv[0])
    parser = OptionParser(usage_msg)

    parser.add_option('-s', '--sflag', dest='sflag', action='store_true', default=False, help='Remove synchronization error.')
    parser.add_option('-o', '--oflag', dest='oflag', action='store_true', default=False, help='Remove outliers.')
    parser.add_option('-t', '--tflag', dest='tflag', action='store_true', default=False, help='Apply the preprocessing on the temperature data.')
    parser.add_option('-j', '--jflag', dest='jflag', action='store_true', default=False, help='Detect jumps in the deformation data.')
    parser.add_option('-n', dest='nh', action='store', type='int', default=12, help='Gaps (in hour) larger than this value will be marked as nan (default 12).')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print('Usage: '+usage_msg)
        sys.exit(0)
    else:  # check projdir
        projdir = args[0]
        if not os.path.isdir(projdir):
            raise FileNotFoundError(projdir)

    Preprocess_static_data(projdir, options)


if __name__ == '__main__':
    print(__script__)
    # print()
    main()
