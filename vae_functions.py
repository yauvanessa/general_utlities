import numpy as np
import torch
import EMtools as DCIP
import utm
import base64
import plotly.express as px
import plotly.graph_objects as go
import dash_daq as daq
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
from vae_classes import *
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from scipy import signal as sig
from os import listdir
from datfiles_lib_parallel import *
from io import BytesIO

def load_training_data(input_dict, cut_off=4136):
    """loads in clean data for model training

    Args:
        input_dict (list): two .npy files with training data and labels 
                        -> = {'learn_data': 'train_master.npy', 'learn_labels': 'labels_master.npy'}
        cut_off (int, optional): change cut off when making call as needed

    Returns:
        _type_: _description_
    """

    try:
        # load learning and test data
        learn_data = np.load(input_dict["learn_data"])
        learn_labels = np.load(input_dict["learn_labels"])
    except Exception:
        print('Could not load training data or labels')

    else:
        # find only the good data
        index_good = learn_labels == 0

        learn_data = learn_data[index_good]
        learn_labels = learn_labels[index_good]

        # data loaders
        learning_data = np.reshape(learn_data, (learn_data.shape[0], 1, learn_data.shape[1]))

        try:
            train_data = DiasTimeSeriesDataset(learning_data[:cut_off], labels=learn_labels[:cut_off])
            test_data = DiasTimeSeriesDataset(learning_data[cut_off:], labels=learn_labels[cut_off:])

            # data loaders
            train_loader = DataLoader(train_data, batch_size=500, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=500, shuffle=True)       
        except Exception:
            print('DIAS pytorch dataset could not be loaded')

    return train_loader, test_loader, train_data, test_data


def training_procedure(autoencoder, train_loader, device, epochs=1000):
    """uses clean training data to train vae model

    """

    try:
        opt = torch.optim.Adam(autoencoder.parameters())
    except Exception:
        print('Pytorch optimization failed, could not train model')

    else:
        # loss_fun = torch.nn.CrossEntropyLoss()
        error_loss = []
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                x, y = data
                y = y.to(device)
                x = x.type(torch.float32)
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat = autoencoder(x).to(device)
                # loss = loss_fun(x, x_hat) + autoencoder.encoder.kl
                loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
                error_loss.append(loss)
                loss.backward()
                opt.step()

    return autoencoder, error_loss


def plot_latent_space(autoencoder, data, device, num_batches=100, saveIt=0):
    """plots the latent/hidden dimensions from the vae
    Saves figure as html if saveIt set to 1. Defaults to 0.
    """
    try:
        app = Dash(__name__)

        for i, (x1, y1) in enumerate(data):
            x1 = x1.type(torch.float32)
            z1 = autoencoder.encoder(x1.to(device))
            z1 = z1.cpu().detach().numpy()

            fig = px.scatter(x=z1[:, 0], y=z1[:, 1], title='latent space')
            fig.update_traces(marker=dict(color=y1, colorscale='sunsetdark', showscale=False))

            if i > num_batches:
                fig.update_traces(marker=dict(showscale=True))
                break
    except Exception:
        print('Could not plot latent space')

    else:
        try:
            app.layout = html.Div([
                dcc.Graph(id='plt_latentspace', figure=fig),
            ])

            # if saveIt == 1:
            #     print('1')
            #     # save_html(fig, 'plt_latentspace.html')
        except Exception:
            print('Could not save figure')

    return fig, app


def plot_reconstructions(testdata, autoencoder, device, saveIt=0):
    """plots reconstructed noise vs training data
    Saves figure as html if saveIt set to 1. Defaults to 0.
    """
    try:
        app = Dash(__name__)
        xc = torch.from_numpy(testdata[3][0]).type(torch.float32).to(device)
        # print(vae(xc).view(1, 1200))

        xc.to(device)

        fig = go.Figure()
        trace1 = go.Line(y=autoencoder(xc).view(1, 1200).cpu().detach().numpy()[0,:], name='VAE')
        trace2 = go.Line(y=xc.cpu().detach().numpy()[0,:], name='TRAIN')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(trace1)
        fig.add_trace(trace2, secondary_y=True)
        # plt.title(f'Reconstruction error: {np.linalg.norm(autoencoder(xc).view(1, 1200).cpu().detach().numpy()[0,:] - xc.cpu().detach().numpy()[0,:])}')
    except Exception:
        print('Could not plot reconstructed noise')

    else:
        try:
            app.layout = html.Div([
                dcc.Graph(id='plt_reconstruction', figure=fig),
            ])
        except Exception:
            print('Could not save figure')

    return fig, app


def load_DCIPtimeseries(path, vae, device):
    """loads in DCIP time series from either a folder OR individual node files. Will try if path given is a directory and will go
    through each item in the directory, otherwise will read in and reconstruct using a single file.

    """
    reconstruction_errors = []
    receiver_dict = {}

    try:
        listdir(path)
    except NotADirectoryError:
        node = path
        receiver_dict[node.split('.')[0]] = {'std': None, 'reconstruction_error': [], 'idx': None, 
                                    'location': None, 'std_raw': None, 'std_ip': None}
#         print(node)
        reconstruction_errors, receiver_dict, xc_data, xt_data = read_nodefile(node, receiver_dict, vae, device)
    else:
        only_files = [f for f in listdir(path)]

        for m in range(len(only_files)):
            
            if only_files[m].split(".")[-1] == "DAT":

                node = path + only_files[m]
                receiver_dict[node.split('.')[0]] = {'std': None, 'reconstruction_error': [], 'idx': None, 
                                                    'location': None, 'std_raw': None, 'std_ip': None}
            
                reconstruction_errors, receiver_dict, xc_data, xt_data = read_nodefile(node, receiver_dict, vae, device)

    return reconstruction_errors, receiver_dict, xc_data, xt_data


def read_nodefile(node, receiver_dict, vae, device):
    """reads in a given node file and reconstructs noise using the vae model

    """

    try:
        fIn = open(node, 'r', encoding="utf8", errors=""'ignore')
        linesFIn = fIn.readlines()
        fIn.close()
    except Exception:
        print('Node file could not be opened')

    else:
        time, data = read_data(linesFIn)

        # get average GPS location from DAT file
        gps_locations = get_average_gps(linesFIn)

        # convert to utm's
        utms = utm.from_latlon(gps_locations[0], gps_locations[1])

        receiver_dict[node.split('.')[0]]['location'] = utms

        # stack the data
        num_half_T = int(np.floor(data.size / 1200))
        new_trim = int(num_half_T * 1200)
        xt_data = data[:new_trim]

        xt_data = np.reshape(xt_data, (1200, num_half_T), order='F')
        receiver_dict[node.split('.')[0]]['std_raw'] = np.std(xt_data[150:280, :]) # TODO check if these will always be consistent
        receiver_dict[node.split('.')[0]]['std_ip'] = np.std(xt_data[310:580, :])

        offset = 0.5

        time_series_reconstruction = []
        reconstruction_error = []

        idx = 0
        for idx in range(xt_data.shape[1]):

            xt_data[:, idx] -= xt_data[:, idx].mean()
            # data /= data.std()

            max_value = np.max(np.abs(xt_data[10:50, idx]))

            # normalise the data
            xt_data[:, idx] = (xt_data[:, idx] / max_value) * 0.2 + offset

            xc = torch.from_numpy(xt_data[:, 0]).type(torch.float32).to(device)

            reconstruction = vae(xc.view(1, 1200)).cpu().detach().numpy()[0, :]

            error = np.linalg.norm(vae(xc.view(1, 1200)).cpu().detach().numpy()[0,:] - xc.view(1, 1200).cpu().detach().numpy()[0, :])

            time_series_reconstruction.append(reconstruction)
            receiver_dict[node.split('.')[0]]['reconstruction_error'].append(error)
            reconstruction_error.append(error)

        xc_data = np.hstack(time_series_reconstruction)

        # print(xc_data.shape, xt_data.shape)

        # noise estimation
        delta_ts = xt_data.flatten(order='F') - xc_data[0, :]

        receiver_dict[node.split('.')[0]]['std'] = np.std(delta_ts)
        # receiver_dict[node.split('.')[0]]['idx'] = m

    return reconstruction_error, receiver_dict, xc_data, xt_data


def plot_histograms(reconstruction_errors, stds, saveIt=0):
    """plots a histogram of errors from vae reconstruction and the standard deviation of the noise estimates
    Saves figure as html if saveIt set to 1. Defaults to 0.

    """
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].hist(np.log(reconstruction_errors), 100)
        axs[0].set_title('a) Reconstruction Errors')
        axs[0].set_xlabel('log(reconstruction error)')

        axs[1].hist([round(decm, 3) for decm in stds], 100) # round off std values
        axs[1].set_title('b) Standard deviation of estimated noise')
        axs[1].set_xlabel('standard deviation')
    except Exception:
        print('Could not plot histogram of vae reconstruction errors')

    else:
        try:
            if saveIt == 1:
                save_html(fig, 'plt_histograms.html')
        except Exception:
            print('Could not save figure')

    return fig, axs


def plot_dist(locations, stds, stds_raw, rec_error, saveIt=0):
    """plots the STD of noise estimate, reconstruction error, and traditional error as a function of node location
    Saves figure as html if saveIt set to 1. Defaults to 0.

    """
    try:
        mpl.rcParams['axes.formatter.useoffset'] = False
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))

        im = axs[0].scatter(np.vstack(locations)[:, 0], np.vstack(locations)[:, 1], c=stds, cmap='Spectral_r')
        axs[0].axis('equal')
        axs[0].set_title('a) Standard deviations of estimated noise')
        fig.colorbar(im, ax=axs[0], label='standard deviation', shrink=0.5, orientation='horizontal')
        axs[0].set_xlabel('northing (m)')
        axs[0].set_ylabel('easting(m)')
        axs[0].tick_params(axis='both', which='major', labelsize=6)

        im1 = axs[1].scatter(np.vstack(locations)[:, 0], np.vstack(locations)[:, 1], c=np.log(rec_error), cmap='Spectral_r')
        axs[1].axis('equal')
        axs[1].set_title('b) VAE Reconstruction error')
        fig.colorbar(im1, ax=axs[1], label='log(error)', shrink=0.5, orientation='horizontal')
        axs[1].set_xlabel('northing (m)')
        axs[1].set_ylabel('easting (m)')
        axs[1].tick_params(axis='both', which='major', labelsize=6)

        im2 = axs[2].scatter(np.vstack(locations)[:, 0], np.vstack(locations)[:, 1], c=np.log(np.array(stds_raw) * 0.05), cmap='Spectral_r')
        axs[2].axis('equal')
        axs[2].set_title('c) Traditional error assignment')
        fig.colorbar(im2, ax=axs[2], label='log(error)', shrink=0.5, orientation='horizontal')
        axs[2].set_xlabel('northing (m)')
        axs[2].set_ylabel('easting (m)')
        axs[2].tick_params(axis='both', which='major', labelsize=6)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    except Exception:
        print('Could not plot standard deviations')

    else:
        try:
            if saveIt == 1:
                save_html(fig, 'plt_reconlocations.html')
        except Exception:
            print('Could not save figure')

    return fig, axs


def plot_gmm(locations, rec_error, saveIt=0):
    """plots the Gaussian Mixture model kmeans clusters
    Saves figure as html if saveIt set to 1. Defaults to 0.
    """
    try:
        gmm = GaussianMixture(n_components=5).fit(locations)
        # Z = -gmm.score_samples(locations)
        # Z = Z.reshape(locations.shape)

        # labels = gmm.predict(X_test)
        print(gmm.weights_)
        # plot the guassian influence
        w_factor = 0.2 / gmm.weights_.max()

        mpl.rcParams['axes.formatter.useoffset'] = False
        fig, ax = plt.subplots(1,1)
        for ii in range(5):
            w = gmm.weights_[ii]
            draw_ellipse(gmm.means_[ii, :], gmm.covariances_[ii, :, :], ax, alpha=w * w_factor* 2)

        im3 = ax.scatter(np.vstack(locations)[:, 0], np.vstack(locations)[:, 1], c=np.log(rec_error), cmap='Spectral_r')
        plt.colorbar(im3, shrink=0.7, label="log(error)")

        ax.plot(gmm.means_[:, 0], gmm.means_[:, 1], 'kD', label='means')
        ax.legend()
        ax.set_title("Gaussian mixture model cluster means")
        ax.axis("equal")
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
    except Exception:
        print('Could not plot figure of Gaussian Mixture model')

    else:
        try:
            if saveIt == 1:
                save_html(fig, 'plt_gaussianmixmod.html')
        except Exception:
            print('Could not save figure')

    return fig, ax


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    try:
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
    except Exception:
        print('Unable to convert covariance to principal axes')
    else:
        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                angle, **kwargs))


def plot_fieldreconstruction(xc_data, xt_data, node, saveIt=0):
    """plot reconstruction of time series for a single node
    Saves figure as html if saveIt set to 1. Defaults to 0.
    """
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(xt_data.flatten(order='F')[:4800], 'k', label='raw time-series')
        axs[0].plot(xc_data[0, :4800], 'g', label='reconstruction')
        axs[0].plot((xt_data.flatten(order='F') - xc_data[0, :])[:4800], 'r', label='$\Delta$ time-series')
        axs[0].set_xlabel('time sample n')
        axs[0].set_ylabel('voltage (mV)')
        axs[0].set_title(f"a) std: {np.std(xt_data.flatten(order='F') - xc_data[0, :])}")
        axs[0].legend(loc="lower right")

        # axs[0, 0].stem(reconstruction_error)
        # axs[0, 0].xlabel('samples n')
        # axs[0, 0].ylabel('reconstruction')
        # axs[0, 0].show()

        f, psd = sig.welch(xt_data.flatten(order='F'), fs=150, window='blackman', nperseg=4056, scaling='spectrum')
        axs[1].loglog(f, psd, 'm', label='raw time-series')
        f, psd = sig.welch(xt_data.flatten(order='F') - xc_data[0, :], fs=150, window='blackman', nperseg=4056, scaling='spectrum')
        axs[1].loglog(f, psd, 'g', label='$\Delta$ time-series')
        f, psd = sig.welch(xc_data[0, :], fs=150, window='blackman', nperseg=4056, scaling='spectrum')
        axs[1].loglog(f, psd, 'b', label='recontruction')
        axs[1].legend()

        axs[1].set_xlabel('frequency (Hz)')
        axs[1].set_ylabel('PSD')
        node1 = node.split('\\')[-1]
        axs[1].set_title(f"b) {node1.rsplit('/')[-1]}")
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.25,
                            hspace=0.4)
    except Exception:
        print('Could not plot time series reconstruction')
        
    else:
        try:
            if saveIt == 1:
                save_html(fig, 'plt_reconfielddata.html')
        except Exception:
            print('Could not save figure')
    
    return fig, axs


def stackDCIPdata(xc_data, xt_data):
    """stacks DCIP data to get voltages and voltage decays
    """
    try:
        timeFrom, timeTo = getTime()

        mid_time = (np.asarray(timeTo) + np.asarray(timeFrom)) / 2
        time_width = np.asarray(timeTo) - np.asarray(timeFrom)

        num_half_T = np.floor(xt_data.flatten(order='F').size / 600)
        new_trim = int(num_half_T * 600)

        xt_data = xt_data.flatten(order='F')[:new_trim]
        xc_data = xc_data[0, :][:new_trim]

    except Exception:
        print('Unable to get time intervals from data')

    else:
        try:
            # start_vp = 50                           # start of Vp calculation (%)
            # end_vp = 90                             # end of Vp calculation (%)
            window = DCIP.createHanningWindow(num_half_T)   # creates filter window
            # window = DCIP.createChebyshevWindow(int(num_half_T), 500)
            # window2 = DCIP.createKaiserWindow(int(num_half_T), 150)

            tHK = DCIP.filterKernel(filtershape=window)     # creates filter kernal
            # tHK2 = DCIP.filterKernel(filtershape=window)     # creates filter kernal
            print("half T: {0} window: {1} Kernel: {2}".format(num_half_T, window.size, tHK.kernel.size))
            # print(xt.size)
            # # eHK = DCIP.ensembleKernal(filtershape=window3,
            # #                           number_half_periods=num_half_T)
            dkernal = DCIP.decayKernel(num_windows=np.asarray(timeTo).size,
                                    window_starts=np.asarray(timeFrom),
                                    window_ends=np.asarray(timeTo),
                                    window_weight=501,
                                    window_overlap=0.99,
                                    output_type="Vs")  # creates decay kernal
        except Exception:
            print('Unable to create kernel filter or decay kernel')

        else:
            stack = tHK * xt_data                               # stack data
            stack2 = tHK * xc_data                               # stack data

            decay = dkernal * (tHK * xt_data)         # calculates the decay
            decay2 = dkernal * (tHK * xc_data)         # calculates the decay

            mx1 = np.sum(decay * time_width) / np.sum(time_width) / (DCIP.getPrimaryVoltage(60, 90, stack) * 1e-3)
            mx2 = np.sum(decay2 * time_width) / np.sum(time_width) / (DCIP.getPrimaryVoltage(60, 90, stack2) * 1e-3)

            print(stack[150:280].mean())
            print(stack2[150:280].mean())
            print(np.std(stack[150:280]))
            print(np.std(stack2[150:280]))
            print(np.std(stack))
            print(np.std(stack2))

    return stack, stack2, decay, decay2, mx1, mx2, mid_time


def getTime():
    """generates the off time intervals 
    *NOTE this function will change if off time changes, current default is 2s*

    Returns:
        _type_: _description_
    """
    timeFrom = [2040., 2060., 2080., 2120., 2160., 2200.,
                2240., 2320., 2400.,
                2480., 2560., 2640.,
                2720., 2800., 2960.,
                3120., 3280., 3440.,
                3600., 3760.]
    timeTo = [2060., 2080., 2120., 2160., 2200., 2240.,
              2320., 2400., 2480., 2560., 2640., 2720.,
              2800., 2960., 3120., 3280., 3440.,
              3600., 3760., 3920.]
    
    return timeFrom, timeTo


def plot_voltagedecay(stack, stack2, decay, decay2, mx1, mx2, mid_time, node, saveIt=0):
    """plots the voltage and voltage decay for a single stacked node
    Saves figure as html if saveIt set to 1. Defaults to 0.
    """
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(stack, label=f" Raw: {DCIP.getPrimaryVoltage(60, 90, stack) :.3} mV")
        axs[0].plot(stack2, 'r', label=f"VAE: {DCIP.getPrimaryVoltage(60, 90, stack2) :.3} mV")
        axs[0].set_xlabel("stack number")
        axs[0].set_ylabel("voltage (mV)")
        node1 = node.split('\\')[-1]
        axs[0].set_title(f"a) {node1.rsplit('/')[-1]}")
        axs[0].legend()

        axs[1].plot(mid_time, decay, label=f"Raw: {mx1: .1f} mV/V")
        axs[1].plot(mid_time, decay2, 'r', label=f"VAE: {mx2: .1f} mV/V")
        axs[1].set_ylabel("voltage (mV)")
        axs[1].set_xlabel("off-time (ms)")
        node1 = node.split('\\')[-1]
        axs[1].set_title(f"b) Voltage decay {node1.rsplit('/')[-1]}")
        axs[1].legend()
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.4)
    except Exception:
        print('Could not plot voltage and decay')

    else:
        try:
            if saveIt == 1:
                save_html(fig, 'plt_voltagedecay.html')
        except Exception:
            print('Could not save figure')
    
    return fig, axs


# def save_html(fig, title):
#     """saves a figure as an .html file with a given title name
#     """

#     try:
#         tmpfile = BytesIO()
#         fig.savefig(tmpfile, format='png')
#         encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
#         html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
#     except Exception:
#         print('Unable to encode png figure as html')

#     else:
#         # write to html
#         with open(title, 'w') as f:
#             f.write(html)
