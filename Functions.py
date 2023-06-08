import os
from datfiles_lib_parallel import *
from os import listdir
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import h5py
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
    cfstr = 'Conversion Factor'
    for n in range(0,17): # number of header option; takes too long to check all
        teststr = linesFIn[n]

        if cfstr in teststr:
            cf = linesFIn[n]
            addto = cf.split(':')
            cfnew = addto[0] + ':-' + addto[1]
            linesFIn[n] = cfnew # rewrite correction

            # overwrite file with fix
            fOut = open(dir, 'w')
            for f in linesFIn:
                fOut.write('%s' % f)
            fOut.close()


def update_resistivity(fIn:str):
    """loads the h5file and calculates + updates the resistivity; rewrites the h5file
    Helpful article: https://archive.epa.gov/esd/archive-geophysics/web/html/resistivity_methods.html

    Args:
        fIn (str): hdf5 file
    """
    # read in h5 file
    f = h5py.File(fIn, 'r+')

    # assign variables from file
    V = f.attrs['voltage']
    I = f.attrs['current']
    #K = f.attrs['geometric_factor']

    # update geometric factor K
    K = calculate_geofactor(f.attrs['receiver1_easting'], f.attrs['receiver1_northing'], f.attrs['receiver1_altitude'], 
                            f.attrs['receiver2_easting'], f.attrs['receiver2_northing'], f.attrs['receiver2_altitude'],
                            f.attrs['transmit_easting'], f.attrs['transmit_northing'], f.attrs['transmit_altitude'],
                            f.attrs['transmit2_easting'], f.attrs['transmit2_northing'], f.attrs['transmit2_altitude'])

    # recalculate resistivity
    res = 2*np.pi*K*(V/I)
    
    # assign new resistivity to file
    f.attrs['apparent_resistivity'] = res
    f.close()    


def calculate_geofactor(Rx1East,Rx1North,Rx1Elev,Rx2East,Rx2North,Rx2Elev,Tx1East,Tx1North,Tx1Elev,Tx2East,Tx2North,Tx2Elev):
    """calculates the geometric factor for resistivity, given rx and tx locations

    Args:
        Rx1East (float): _description_
        Rx1North (float): _description_
        Rx1Elev (float): _description_
        Rx2East (float): _description_
        Rx2North (float): _description_
        Rx2Elev (float): _description_
        Tx1East (float): _description_
        Tx1North (float): _description_
        Tx1Elev (float): _description_
        Tx2East (float): _description_
        Tx2North (float): _description_
        Tx2Elev (float): _description_

    Returns:
        float: recalculated geometric factor
    """
    r1 = ((Rx1East - Tx1East)**2 +
        (Rx1North - Tx1North)**2 +
        (Rx1Elev - Tx1Elev)**2)**0.5
    r2 = ((Rx2East - Tx1East)**2 +
        (Rx2North - Tx1North)**2 +
        (Rx2Elev - Tx1Elev)**2)**0.5
    r3 = ((Rx1East - Tx2East)**2 +
        (Rx1North - Tx2North)**2 +
        (Rx1Elev - Tx2Elev)**2)**0.5
    r4 = ((Rx2East - Tx2East)**2 +
        (Rx2North - Tx2North)**2 +
        (Rx2Elev - Tx2Elev)**2)**0.5

    K_new = 1 / ((1 / r1 - 1 / r2) - (1 / r3 - 1 / r4))

    return K_new


def decay_fit(fIn:str):
    """fits decay curve to a stretched exponential function; returns fit parameters

    Args:
        fIn (str): hdf5 file
    """

    # read in h5 file
    f = h5py.File(fIn, 'r+')

    t = f['vs_window_centers']
    t = t[:]/1000 # get from ms to s
    Vs = f['decay']
    # plt.scatter(t,Vs)

    # fit decay curve
    popt, _ = curve_fit(exponential_decay_estimate, t, Vs, p0=(0.5,0.5,0.5)) # random initial guesses req to get ok fit
    a, b, c = popt # get fit parameters

    # get y data from fit
    exp_fit = exponential_decay_estimate(t, a, b, c)
    # plt.scatter(t,exp_fit)

    # calculate std of fit
    std = np.std(Vs - exp_fit)

    # write parameters to file
    f.attrs['decay_fit_a'] = a
    f.attrs['decay_fit_b'] = b
    f.attrs['decay_fit_c'] = c
    f.attrs['decay_fit_std'] = std
    f.close()


def exponential_decay_estimate(x, a, b, c):
    """stretched exponential function

    Args:
        x (array): x data
        a (float): function parameter
        b (float): function parameter
        c (float): function parameter

    Returns:
        array: y data
    """
    return a * np.exp(-x ** b) + c