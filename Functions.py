import os
from datfiles_lib_parallel import *
from os import listdir
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def replace_locations(sessionID:int, RXfile:str, TXfile:str, dir:str, datInfo=[]):
    """function for replacing reciever GPS with more accurate location; loops through multiple nodes

    Args:
        sessionID (int): data ID number
        RXfile (str): RX file
        TXfile (str): TX file
        dir (str): data directory to raw data folders
        datInfo (list): list of information (Unit ID, MEM number, Relay state, if current of potential)
    """
    # select location from Rx OR Tx file
    if len(datInfo) != 0 and datInfo[3] == "V":
        # load Tx file from csv
        txfile = pd.read_csv(os.path.join(dir, TXfile))
        file = txfile.to_numpy()
        tx_utm = file[:,[2,3]]
        # fit nearest neighbour to tx pts
        nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree',p=2).fit(tx_utm)
    else: 
        # use Rx file ("C"), even if None
        # load Rx file from csv
        rxfile = pd.read_csv(os.path.join(dir, RXfile))
        file = rxfile.to_numpy()
        rx_utm = file[:,[2,3]]
        # fit nearest neighbour to rx pts
        nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree',p=2).fit(rx_utm)     

    # load in data file gps pts
    all_folds = [x[0] for x in os.walk(dir)]
    dat_folds = all_folds[1:]

    for m in range(len(dat_folds)):
        # get files within data folder
        only_files = [f for f in listdir(dat_folds[m])]
        for idx in range(len(only_files)):
                node = os.path.join(dat_folds[m], '', only_files[idx])
                fIn = open(node, 'r', encoding="utf8", errors=""'ignore')
                linesFIn = fIn.readlines()
                fIn.close()

                # time, data = read_data(linesFIn)
                gps = get_average_gps(linesFIn)

                # convert gps to UTM
                gps_UTM = utm.from_latlon(gps[0],gps[1])
                gps_UTM_pt = np.array([gps_UTM[0],gps_UTM[1]])

                # find closest neighbour to gps pt
                dists, inds = nbrs.kneighbors([gps_UTM_pt])

                # get closest gps pt from rx file
                closest = file[inds[0,0]]

                # check accuracy
                # print(f'diff1= ',gps_UTM_pt[0]-closest[2],'diff2= ',gps_UTM_pt[1]-closest[3])

                # convert closest pt back to lat lon # TODO check error on this, maybe stick to UTM -> yes.
                # closest_latlon = utm.to_latlon(closest[2],closest[3],gps_UTM[2],gps_UTM[3])

                # force new location to file (in UTM)
                force_file(node,closest[0],closest[1],closest[4],False,1) # TODO placeholder values


def force_file(path:str, easting:float, northing:float, elevation:float, flag_new:bool, sid: int) -> None:
    """forces node files to a specified easting, northing, elevation

    Args:
        path (str): raw directory ID
        easting (float): known latitude
        northing (float): known longitude
        elevation (float): elevation
        flag_new (bool): _description_
        sid (int): session bucket ID
    """


def polarity_correction(dir:str):
    """changes polarity conversion factor from positive to negative

    Args:
        dir (str): data file (ascii)
    """
     
     # read in ASCII file
    fIn = open(dir, 'r', encoding="utf8", errors='ignore')
    linesFIn = fIn.readlines()
    print(fIn.read())
    fIn.close()

    # get conversion factor from header
    cf = linesFIn[5]
    addto = cf.split(':')
    cfnew = addto[0] + ':-' + addto[1]
    linesFIn[5] = cfnew # rewrite correction

    # overwrite file with fix
    fOut = open(dir, 'w')
    for f in linesFIn:
         fOut.write('%s' % f)
    fOut.close()

def IP_resistivity():
    """calculate resistivity 
    """