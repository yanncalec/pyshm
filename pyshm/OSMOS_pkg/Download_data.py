"""Download or update database from the server of OSMOS."""

# __all__ = ["Download_data"]

import sys, os
import requests, json, pandas, colorama
from . import OSMOS

class Options:
    PID=None
    force=False
    verbose=False

def Download_data(datadir, options=None):
    """
    datadir : the directory of the whole database
    PID : Keyid of the project to be downloaded, a number
    verbose : if True print message
    """

    if options is None:  # use default value for options
        options = Options()

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

    # Here is the token to use for next requests
    token = response['data']["tokenStore"]

    # get PROJECT to analyse
    payload = {"token": token, "action": "getlistprojectstoanalyse"}
    r = session.post("https://client.osmos-group.com/server/application.php", data=json.dumps(payload))
    response = json.loads(r.text)
    assert(response['result'] == 'SUCCESS')  # check the response from the server

    ListProject = response['data']  # List of projects

    # Save in a dictionary with project keyid as keys
    DicProject = {}
    for p in ListProject:
        DicProject[p['projectkeyid']] = p

    # Informations of the available projects
    toto = pandas.DataFrame(ListProject)
    Pinfo = toto[['projectkeyid', 'name', 'level', 'start', 'end']].sort_values('projectkeyid').reset_index(drop=True)  # Information of projects
    Pinfo.to_excel(os.path.join(datadir, 'info.xlsx'))  # save (overwrite) the information in a Excel file

    # List of PID to be updated
    if options.PID is not None:  # PID given
        if options.PID in DicProject.keys():
            ListPID = [options.PID]
        else:
            raise KeyError('PID {} not found'.format(options.PID))
    else:  # PID not given
        ListPID = list(DicProject.keys())  # use the whole list of PIDs
        ListPID.sort()

    # Download data from OSMOS's server
    newdata_list = []  # PIDs with successful fetching of new data
    failedPID_list = []  # PID with failed response from server or empty project

    for pid in ListPID:
        datadir1 = os.path.join(datadir, '{:03}'.format(pid)) # output directory, which will be created if necessary by the function OSMOS.update_LIRIS_data_by_project()

        info = DicProject[pid]
        if options.verbose:
            print('\n---------Fetching project {}---------'.format(pid))
            print('Name: {}'.format(info['name']))
            print('Level: {}'.format(info['level']))
            print('Start: {}'.format(info['start']))
            print('End: {}'.format(info['end']))

        try:
            # new data are saved in sub-folders named by the locations' keyid
            nflag = OSMOS.update_LIRIS_data_by_project(token, session, pid, datadir1, verbose=options.verbose)

            if nflag:
                newdata_list.append(pid)

                # Save the information of the project in a text file
                fname = os.path.join(datadir1, 'info.txt') # output file name
                if not os.path.isfile(fname):  #
                    with open(fname, 'w') as fp:
                        fp.writelines('name : {}\n'.format(info['name']))
                        fp.writelines('level : {}\n'.format(info['level']))
                        fp.writelines('start : {}\n'.format(info['start']))
                        fp.writelines('end : {}\n'.format(info['end']))

            if nflag or options.force:
                # update the assembled pandas data file
                OSMOS.assemble_to_pandas(datadir1)

        except Exception as msg:
            failedPID_list.append(pid)
            print(colorama.Fore.RED + 'Error: ', msg)
            print(colorama.Style.RESET_ALL)

    if options.verbose:
        print('\nProject(s) with new data:', newdata_list)
        print('Failed project(s):', failedPID_list)

    return newdata_list, failedPID_list
