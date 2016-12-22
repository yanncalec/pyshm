#!/usr/bin/env python

"""Download or update the local database from the server of OSMOS.
"""

class Options:
    PID=None  # Key ID of the project to be downloaded, a number
    force=False  # Force assembling of data
    delete=False  # Delete failed projects
    endtime=None  # fetch data til this ending time
    verbose=False  # print message
    plot=False  # plot data


def Download_data(dbdir, options):
    """Download or update the local database from the server of OSMOS.

    Args:
        dbdir (string): the directory of the whole database
        options: object including all options, e.g., returned by parser.parse_args()
    """

    import requests, json, pandas, colorama, dateutil
    import sys, os
    # try:
    #     sys.path.index('/dir/path') # Or os.getcwd() for this directory
    # except ValueError:
    #     sys.path.append('/dir/path') # Or os.getcwd() for this directory

    from pyshm import OSMOS

    # disable certificate check
    requests.packages.urllib3.disable_warnings()

    # open the session
    session = requests.session()

    # get the token
    payload = {"token":"null","action":"loginClient","data":{"email":"be@osmos-group.com","password":"osmos"}}
    r = session.post("https://client.osmos-group.com/server/application.php", data=json.dumps(payload))
    response = json.loads(r.text)
    assert(response['result'] == 'SUCCESS')  # check the response from the server
    # print(session.cookies)
    # print(response)

    # token to use for next requests
    token = response['data']["tokenStore"]

    # get informations of projects
    payload = {"token": token, "action": "getlistprojectstoanalyse"}
    r = session.post("https://client.osmos-group.com/server/application.php", data=json.dumps(payload))
    response = json.loads(r.text)
    assert(response['result'] == 'SUCCESS')  # check the response from the server

    ListProject = response['data']  # List of projects

    # Save in a dictionary with project keyid as keys
    DicProject = {p['projectkeyid']:p for p in ListProject}
    # which is the pythonic way for the following:
    # DicProject = {}
    # for p in ListProject:
    #     DicProject[p['projectkeyid']] = p

    # Informations of the available projects
    toto = pandas.DataFrame(ListProject)
    Pinfo = toto[['projectkeyid', 'name', 'level', 'start', 'end']].sort_values('projectkeyid').reset_index(drop=True)  # Information of projects
    Pinfo.to_excel(os.path.join(dbdir, 'info.xlsx'))  # save (overwrite) the information in a Excel file

    # List of PID to be updated
    if options.PID is not None:  # PID given
        if options.PID in DicProject.keys():  # check that the given PID is valid
            ListPID = [options.PID]
        else:
            raise KeyError('PID {} not found'.format(options.PID))
    else:  # PID not given
        ListPID = list(DicProject.keys())  # use the whole list of PIDs
        ListPID.sort()

    # Download data from OSMOS's server
    newdata_list = []  # PIDs with successful fetching of new data
    failedPID_list = []  # PID with failed response from server or empty project

    for pid in ListPID:  # iteration on the PID
        projdir = os.path.join(dbdir, '{:03}'.format(pid)) # output directory

        info = DicProject[pid]
        if options.verbose:
            print('\n---------Fetching project {}---------'.format(pid))
            print('Name: {}'.format(info['name']))
            print('Level: {}'.format(info['level']))
            print('Start: {}'.format(info['start']))
            print('End: {}'.format(info['end']))

        try:
            # new data are saved in sub-folders named by the locations' keyid
            nflag, Liris = OSMOS.update_LIRIS_data_by_project(token, session, pid, projdir, endtime=options.endtime, verbose=options.verbose)

            if nflag:  # successfully fetched new data
                newdata_list.append(pid)
            if nflag or options.force:
                    # update the assembled pandas data file
                    OSMOS.assemble_to_pandas(projdir)  # assemble to Raw.pkl in the project folder

            # Save the information of the project in a text file
            fname = os.path.join(projdir, 'info.txt') # output file name
            if not os.path.isfile(fname):  # creat only if file does not exist
                with open(fname, 'w') as fp:
                    fp.writelines('---------------------------\n')
                    fp.writelines('Information on the project:\n')
                    fp.writelines('---------------------------\n')
                    fp.writelines('name : {}\n'.format(info['name']))
                    fp.writelines('level : {}\n'.format(info['level']))
                    fp.writelines('start : {}\n'.format(info['start']))
                    fp.writelines('end : {}\n'.format(info['end']))
                    fp.writelines('---------------------------\n')
                    fp.writelines('Information on the sensors:\n')
                    fp.writelines('---------------------------\n')
                    for liris in Liris:
                        msg = 'Location ID: {}, Location name: {}, LIRIS ID: {}, LIRIS name:{}\n'.format(liris.locationkeyid, liris.locationname, liris.liriskeyid, liris.name)
                        fp.writelines(msg)

            if options.plot:
                Rdata, Sdata, Ddata, Locations = OSMOS.load_raw_data(os.path.join(projdir, 'Raw.pkl'))
                figdir = os.path.join(projdir, 'figures', 'Raw')
                try:
                    os.makedirs(figdir)
                except:
                    pass
                mpld3_plot(figdir, Rdata, Sdata, Ddata)
        except Exception as msg:
            failedPID_list.append(pid)
            print(colorama.Fore.RED + 'Error: ', msg)
            print(colorama.Style.RESET_ALL)
            # raise Exception

    if options.verbose:
        print('\nProject(s) with new data:', newdata_list)
        print('Failed project(s):', failedPID_list)

    if options.delete:
        from shutil import rmtree  # high-level file operations

        try:
            for pid in failedPID_list:
                projdir = os.path.join(dbdir, '{:03}'.format(pid))
                if os.path.isdir(projdir): # if the project folder exists
                    ans = None
                    while ans not in ['Y', 'N', '']:
                        ans = input('Delete the folder of the project {} and all its contents [y/N]?'.format(pid)).capitalize()
                    if ans=='Y':
                        try:
                            rmtree(projdir)
                        except Exception as msg:
                            print(msg)
                            pass
        except KeyboardInterrupt as e:
            print()
            pass

    return newdata_list, failedPID_list


def mpld3_plot(figdir, Rdata, Sdata, Ddata):
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

    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel('Temperature')
    axes[1].set_ylabel('Elongation')
    plt.tight_layout()

    mpld3.save_html(fig, os.path.join(figdir_html, 'All_static.html'))
    fig.savefig(os.path.join(figdir_pdf, 'All_static.pdf'))
    plt.close(fig)

    # plot all data of each sensor in separated files

    for loc, val in Rdata.items():
        fig, axes = plt.subplots(2,1,figsize=(20,10), sharex=True)
        Xt, Yt = val['Temperature'], val['Elongation']
        axes[0].plot(Xt,'r')#, label='{}'.format(loc))
        axes[1].plot(Yt,'b')
        axes[0].set_ylabel('Temperature')
        axes[1].set_ylabel('Elongation')

        # highlight dynamic events
        for v in Ddata[loc]:
            # axes[1].axvspan(v.index[0], v.index[-1], color='r', alpha=0.3)
            axes[1].plot(v, 'r')  # down-sampling dynamic events

        plt.tight_layout()
        mpld3.save_html(fig, os.path.join(figdir_html, '{}.html'.format(loc)))
        fig.savefig(os.path.join(figdir_pdf, '{}.pdf'.format(loc)))
        plt.close(fig)


__all__ = ['Download_data', 'Options']

__script__ = __doc__

import sys, os, argparse

def main():
    usage_msg = '%(prog)s [options] <dbdir>'

    parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)

    parser.add_argument('dbdir', help='directory of the OSMOS database')
    parser.add_argument('-p', '--PID', dest='PID', type=int, default=None, help='Project Key ID. If not given all projects presented on the remote server will be processed.')
    parser.add_argument('--end', dest='endtime', type=str, default=None, help='Fetch data til this time. If not given fetch data til today.')
    # parser.add_option('-a', '--assemble', dest='assemble', action='store_true', default=False, help='Assemble all pkl files of different Liris of the same PID into a single pkl file named \'Raw_latest.pkl\'.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', default=False, help='Force to assembling data of all sensors into a single file (even no new data are fetched).')
    parser.add_argument('-d', '--delete', dest='delete', action='store_true', default=False, help='Delete failed projects from database.')
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='Plot data of all sensors in the subfolder \'figures\' (could be memory-consuming).')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print message.')

    options = parser.parse_args()

    # if len(args) < 1:
    #     print('Usage: '+usage_msg)
    #     print(parm_msg)
    #     sys.exit(0)
    # else:  # check dbdir
    #     dbdir = args[0]

    # check the database directory
    if not os.path.isdir(options.dbdir):
        raise FileNotFoundError(options.dbdir)

    # core function
    Download_data(options.dbdir, options)


if __name__ == '__main__':
    main()
