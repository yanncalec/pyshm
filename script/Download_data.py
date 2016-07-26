#!/usr/bin/env python

import sys
import os
# import glob
from optparse import OptionParser       # command line arguments parser
import requests
import json
# import datetime
# import dateutil
# from collections import namedtuple
# import warnings
# import itertools
# import copy
import pandas
import colorama

from OSMOS import OSMOS

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hide annoying trace back message
# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

__script__ = 'Download and update data from the server of OSMOS.'


def main():
    usage_msg = '{} [options] data_directory'.format(sys.argv[0])
    # example_msg = 'Example: '

    parser = OptionParser(usage_msg)

    parser.add_option('-p', '--PID', dest='PID', type='int', default=None, help='Project Key ID. If not given all projects presented in the destination data directory will be processed.')
    # parser.add_option('-a', '--assemble', dest='assemble', action='store_true', default=False, help='Assemble all pkl files of different Liris of the same PID into a single pkl file named \'Raw_latest.pkl\'.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print message.')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        print(usage_msg)
        sys.exit(0)
    else:  # check datadir
        datadir = args[0]
        if not os.path.isdir(datadir):
            raise FileNotFoundError(datadir)

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
    toto = toto[['projectkeyid', 'name', 'level', 'start', 'end']].sort_values('projectkeyid').reset_index(drop=True)
    toto.to_excel(datadir+'/info.xlsx')

    # List of PID to be updated
    if options.PID is not None:  # PID given
        if options.PID in DicProject.keys():
            ListPID = [options.PID]
        else:
            raise KeyError('PID {} not found'.format(options.PID))
    else:  # PID not given
        ListPID = list(DicProject.keys())  # list of PIDs
        ListPID.sort()

    # Download data from OSMOS's server
    for PID in ListPID:
        info = DicProject[PID]  # information on the project

        if options.verbose:
            print('\n---------Updating Project {}---------'.format(PID))
            print('Name: {}'.format(info['name']))
            print('Level: {}'.format(info['level']))
            print('Start: {}'.format(info['start']))
            print('End: {}'.format(info['end']))

        try:
            datadir1 = datadir + '/{:03}/'.format(PID) # output directory

            # new data are saved in the sub-directory named by the location
            OSMOS.update_LIRIS_data_by_project(token, session, PID, datadir1, verbose=options.verbose)

            fname = datadir1 + '/info.txt'
            if not os.path.isfile(fname):
                with open(fname, 'w') as fp:
                    fp.writelines('name : {}\n'.format(info['name']))
                    fp.writelines('level : {}\n'.format(info['level']))
                    fp.writelines('start : {}\n'.format(info['start']))
                    fp.writelines('end : {}\n'.format(info['end']))

            # update the assembled pandas data file
            OSMOS.assemble_to_pandas(datadir1)

        except Exception as msg:
            print(colorama.Fore.RED + 'Warning: ', msg)
            print(colorama.Style.RESET_ALL)

if __name__ == '__main__':
    print(__script__)
    print()
    main()
