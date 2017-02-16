#!/usr/bin/env python

"""Download or update the local database.
"""

import os
import argparse
from pyshm.script import examplestyle, warningstyle

class Options:
    PID=None  # Key ID of the project to be downloaded, a number
    force=False  # Force assembling of data
    delete=False  # Delete failed projects (interactive)
    endtime=None  # fetch data til this ending time
    verbose=False  # print message


def Download_data(dbdir, options):
    """Download or update the local database from the server of OSMOS.

    Args:
        dbdir (string): the directory of the whole database
        options: object including all options, e.g., returned by parser.parse_args()
    """

    import requests, json, pandas, colorama, dateutil, os
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
    if options.info:
        toto = pandas.DataFrame(ListProject)
        Pinfo = toto[['projectkeyid', 'name', 'level', 'start', 'end']].sort_values('projectkeyid').reset_index(drop=True)  # Information of projects
        fname_info = os.path.join(dbdir, 'info.xlsx')
        Pinfo.to_excel(fname_info)  # save (overwrite) the information in a Excel file
        print('List of available projects saved in {}'.format(fname_info))
        raise SystemExit

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
            print('Name: {}'.format(info['name']).encode('utf-8'))
            print('Level: {}'.format(info['level']).encode('utf-8'))
            print('Start: {}'.format(info['start']).encode('utf-8'))
            print('End: {}'.format(info['end']).encode('utf-8'))

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


__all__ = ['Download_data', 'Options']

__script__ = __doc__

__warning__ = "Warning:" + warningstyle("\n It is advised AGAINST manual modifications of the local database (e.g., insert, delete or rename files or folders in the database directory), since this may lead to its dysfunction.")

examples = []
examples.append(["%(prog)s --info DBDIR", "Save the list of available projects into a file named info.xlsx in the local database directory DBDIR and exit."])
examples.append(["%(prog)s -p 24 -v DBDIR", "Download or update the project of PID 24 in the directory DBDIR (this will create a project subfolder 024 under DBDIR and the final output is a file named Raw.pkl in there) and print messages."])
examples.append(["%(prog)s -v DBDIR", "Download or update all available projects in the directory DBDIR."])
__example__ = "Some examples of use (change the path seperator '/' to '\\' on Windows platform):" + "".join([examplestyle(x) for x in examples])


def main():
    # usage_msg = '%(prog)s [options] <dbdir>'
    # parser = argparse.ArgumentParser(description=__script__, usage=usage_msg)
    parser = argparse.ArgumentParser(description=__script__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__warning__ + "\n\n" + __example__)

    parser.add_argument('dbdir', help='directory of the local OSMOS database')
    parser.add_argument('-p', '--PID', dest='PID', type=int, default=None, help='Project Key ID. By default all projects presented on the remote server will be processed.', metavar='integer')
    parser.add_argument('-e', '--end', dest='endtime', type=str, default=None, help='Fetch data til this time. By default all data til today will be fetched.', metavar='string')
    # parser.add_option('-a', '--assemble', dest='assemble', action='store_true', default=False, help='Assemble all pkl files of different Liris of the same PID into a single pkl file named \'Raw_latest.pkl\'.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', default=False, help='Force to assembling data of all sensors into a single file (even no new data are fetched).')
    parser.add_argument('-d', '--delete', dest='delete', action='store_true', default=False, help='Delete failed projects from database.')
    parser.add_argument('--info', dest='info', action='store_true', default=False, help='Save the list of available projects and exit.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0, help='Print messages.')

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
        # # create the directory if not existed
        # try:
        #     os.makedirs(options.dbdir)
        # except OSError:
        #     pass

    # core function
    Download_data(options.dbdir, options)


if __name__ == '__main__':
    main()
