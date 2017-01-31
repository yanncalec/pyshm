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
    plot=False  # Plot data of all sensors in the subfolder \'figures\'


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
    Data0, *_ = OSMOS.load_static_data(fname)
    with open(os.path.join(projdir,'update.txt'), 'w') as fp:
        fp.writelines('Location key IDs:{}\n'.format(Locations))
        fp.writelines('---------------------------------\n')
        fp.writelines('Period of the newest measurement:\n')
        fp.writelines('---------------------------------\n')
        for loc in Locations:
            fp.writelines('Location ID {}: from {} to {}\n'.format(loc, Data0[loc].index[0], Data0[loc].index[-1]))
        t0, t1 = OSMOS.common_time_range(Data0)
        fp.writelines('Common period: from {} to {}\n'.format(t0, t1))

    
    if options.plot:
        Data, *_ = OSMOS.load_static_data(fname)
        figdir = os.path.join(projdir, 'figures', 'Static')
        try:
            os.makedirs(figdir)
        except:
            pass
        mpld3_plot(figdir, Data)


def mpld3_plot(figdir, Sdata):
    """Plot in interactive html and pdf files using mpld3 package.
    """
    import matplotlib
    # matplotlib.use("qt5agg")
    import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    import mpld3
    plt.style.use('ggplot')

    # plot static data of all sensors in a single file
    figdir_html = os.path.join(figdir, 'html')
    figdir_pdf = os.path.join(figdir, 'pdf')
    try:
        os.makedirs(figdir_html)
        os.makedirs(figdir_pdf)
    except:
        pass

    fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)

    for loc, val in Sdata.items():
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt, label='{}'.format(loc))
        axes[1].plot(Yt, label='{}'.format(loc))

    # axes[0].legend()
    # axes[1].legend()
    axes[0].legend(fancybox=True, framealpha=0.5)
    axes[1].legend(fancybox=True, framealpha=0.5)
    axes[0].set_ylabel('Temperature')
    axes[1].set_ylabel('Elongation')
    plt.tight_layout()

    mpld3.save_html(fig, os.path.join(figdir_html, 'All_static.html'))
    fig.savefig(os.path.join(figdir_pdf, 'All_static.pdf'))
    plt.close(fig)

    # plot all data of each sensor in separated files

    for loc, val in Sdata.items():
        fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt,'r')#, label='{}'.format(loc))
        axes[1].plot(Yt,'b')
        axes[0].set_ylabel('Temperature')
        axes[1].set_ylabel('Elongation')
        plt.tight_layout()

        mpld3.save_html(fig, os.path.join(figdir_html, '{}.html'.format(loc)))
        fig.savefig(os.path.join(figdir_pdf, '{}.pdf'.format(loc)))
        plt.close(fig)


__all__ =["Preprocess_static_data", "Options"]

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n  Run this script everytime after a sucessful updating of the local database with the script osmos_download. In most cases the optional preprocessings listed above are not necessary and it is recommended to run this script with default parameters.")

examples = []
examples.append(["%(prog)s -v --plot DBDIR/153", "Apply preprocessing with default parameters on the project of PID 153 (the project lied in the database directory DBDIR), plot the static data in a subfolder named figures/Static and print messages."])
examples.append(["%(prog)s --sflag -v --plot DBDIR/036", "Apply preprocessing by removing syncrhonisation error on the project of PID 36."])
__example__ = "Some examples of use:" + "".join([examplestyle(x) for x in examples])


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
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='Plot data of all sensors in the subfolder \'figures\' (could be memory-consuming).')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=False, help='Print messages.')

    options = parser.parse_args()

    if not os.path.isdir(options.projdir):
        raise FileNotFoundError(options.projdir)

    Preprocess_static_data(options.projdir, options)


if __name__ == '__main__':
    main()
