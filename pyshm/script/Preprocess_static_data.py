#!/usr/bin/env python

"""Apply preprocessing and assemble processed data.
"""

import os, argparse
from pyshm.script import examplestyle, warningstyle

class Options:
    sflag=False  # Remove synchronization error
    oflag=False  # Remove outliers
    tflag=False  # Apply the preprocessing on the temperature data
    jflag=False  # Detect jumps in elongation
    nh=12  # Gaps (in hour) larger than this value will be marked as nan
    verbose=False  # Print message


def Preprocess_static_data(projdir, options):
    """Apply preprocessing and assemble processed data.

    Args:
        projdir (string): the directory of a project in the database
        options: object including all options, e.g., returned by parser.parse_args()
    Outputs:
        A file named Processed.pkl in projdir.
    """

    import pickle
    import colorama
    # import glob
    # import copy
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

    # if os.path.isfile(os.path.join(projdir,'info.txt')):
    # Data0, *_ = OSMOS.load_static_data(fname)
    with open(os.path.join(projdir,'update.txt'), 'w') as fp:
        fp.writelines('Location key IDs:{}\n'.format(Locations))
        fp.writelines('---------------------------------\n')
        fp.writelines('Period of the newest measurement:\n')
        fp.writelines('---------------------------------\n')
        for loc in Locations:
            fp.writelines('Location ID {}: from {} to {}\n'.format(loc, Sdata[loc].index[0], Sdata[loc].index[-1]))
        t0, t1 = OSMOS.common_time_range(Sdata)
        fp.writelines('Common period: from {} to {}\n'.format(t0, t1))


__all__ =["Preprocess_static_data", "Options"]

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n  Run this script everytime after a sucessful updating of the local database with the script osmos_download. In most cases the optional preprocessings listed above are not necessary and it is recommended to run this script with default parameters.")

examples = []
examples.append(["%(prog)s -v DBDIR/153", "Apply preprocessing with default parameters on the project of PID 153 (the project lied in the database directory DBDIR), plot the static data in a subfolder named figures/Static and print messages."])
examples.append(["%(prog)s --sflag -v DBDIR/036", "Apply preprocessing by removing syncrhonisation error on the project of PID 36."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s [options] <projdir>'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__warning__ + "\n\n" + __example__)

    parser.add_argument('projdir', help='directory of a project in the database')
    parser.add_argument('-s', '--sflag', dest='sflag', action='store_true', default=False, help='Remove synchronization error.')
    parser.add_argument('-o', '--oflag', dest='oflag', action='store_true', default=False, help='Remove outliers.')
    parser.add_argument('-t', '--tflag', dest='tflag', action='store_true', default=False, help='Apply the preprocessing on the temperature data.')
    parser.add_argument('-j', '--jflag', dest='jflag', action='store_true', default=False, help='Detect jumps in the deformation data.')
    parser.add_argument('-n', dest='nh', action='store', type=int, default=12, help='Gaps (in hour) larger than this value will be marked as nan (default 12).')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=False, help='Print messages.')

    options = parser.parse_args()

    if not os.path.isdir(options.projdir):
        raise FileNotFoundError(options.projdir)

    Preprocess_static_data(options.projdir, options)


if __name__ == '__main__':
    main()
