import cmocean
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import cmocean.cm as cm

cmaps = {
    'w':  {'cm': 'seismic',   'label': 'vertical velocity [m/s]'},
    'ws': {'cm': 'gist_stern_r',              'label': 'windspeed [m/s]'},
    'wd': {'cm': cmocean.cm.phase,   'label': 'wind direction [deg]'},
    'pt': {'cm': cmocean.cm.thermal, 'label': 'potential temperature [C]'},
    't': {'cm': cmocean.cm.thermal, 'label': 'temperature [C]'},
    'q':  {'cm': cmocean.cm.haline_r,  'label': 'q [g/kg]'},
    'dp': {'cm': cmocean.cm.haline_r,  'label': 'dewpoint [C]'},
    'rh': {'cm': cmocean.cm.haline_r,  'label': 'RH [%]'},
    'std': {'cm': cmocean.cm.thermal,  'label': 'Standard Deviation'}
}

def corr_plot(data, heights):
    """
    A handy funtion to create a correlation plot for temperature and mixing ratio
    """
    
    nz = len(heights)
    x, y = np.meshgrid(heights, heights)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    print(np.max(data[:nz, :nz]))
    m1 = ax1.contourf(x, y, data[:nz, :nz], levels=np.arange(-1, 1.05, .05), cmap=cm.balance, vmin=-1, vmax=1)
    ax1.set_title("Temperature Correlation")
    fig.colorbar(m1, ax=ax1)
     
    m2 = ax2.contourf(x, y, data[nz:2*nz, nz:2*nz], levels=np.arange(-1, 1.05, .05), cmap=cm.balance, vmin=-1, vmax=1)

    ax2.set_title("Moisture Correlation")
    fig.colorbar(m2, ax=ax2)
    return fig
    
    
def cov2corr(A):
    """
    Utility function to convert a covariance matrix to a correlation matrix
    """
    d = np.sqrt(A.diagonal())
    return ((A.T/d).T)/d


def timeheight(time, height, data, field, ax, datemin=None, datemax=None,
                datamin=None, datamax=None, zmin=None, zmax=None, cmap=None, **kwargs):
    '''
    Produces a time height plot of a 2-D field
    :param time: Array of times (1-D or 2-D but must have same dimenstions as height)
    :param height: Array of heights (1-D or 2-D but must have same dimensions as time)
    :param data: Array of the data to plot (2-D)
    :param field: Field being plotted. Currently supported:
        'w': Vertical Velocity
        'ws': Wind Speed
        'wd': Wind Direction
        'pt': Potential Temperature
        'q':  Specific Humidity
        'dp': Dewpoint
        'rh': Relative Humidity
        'std': Standard Deviation
    :param ax: Axis to plot the data to
    :param datemin: Datetime object
    :param datemax: Datetime object
    :param datamin: Minimum value of data to plot
    :param datamax: Maximum value of data to plot
    :param zmin: Minimum height to plot
    :param zmax: Maximum height to plot
    :return:
    '''

    # Get the colormap and label of the data
    if cmap is None:
        cm, cb_label = cmaps[field]['cm'], cmaps[field]['label']
    else:
        cm, cb_label = cmap, cmaps[field]['label']

    # Convert the dates to matplolib format if not done already
    if time.ndim == 1 and height.ndim == 1:
        time = mdates.date2num(time)
        time, height = np.meshgrid(time, height)

    # Create the plot
    c = ax.pcolormesh(time, height, data, vmin=datamin, vmax=datamax, cmap=cm, **kwargs)

    # Format the colorbar
    # c.cmap.set_bad('grey', 1.0)
    cb = plt.colorbar(c, ax=ax)
    cb.set_label(cb_label)

    # Format the limits
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if zmin is not None and zmax is not None:
        ax.set_ylim(zmin, zmax)
    if datemin is not None and datemax is not None:
        ax.set_xlim(mdates.date2num(np.array([datemin, datemax])))

    # Set the labels
    ax.set_ylabel('Height [m]')
    ax.set_xlabel('Time [UTC]')

    return ax

def convolve_tropoe_akern(akern, Xa, t, wv, nlevels=55):

    # Concatenate the temperature and wv profiles

    Xsonde = np.concatenate((t, wv))

    # Make sure that the arrays from AERIoe don't have any of the trace gas or cloud information
    Xa = Xa[:int(nlevels*2)]
    akern = akern[:int(2*nlevels), :int(2*nlevels)]

    # Make sure everything is the correct size
    assert len(Xsonde) == len(Xa)

    Xsmooth = np.matmul(akern, (Xsonde - Xa)) + Xa

    smoothed_t = Xsmooth[:nlevels]
    smoothed_wv = Xsmooth[nlevels:int(2*nlevels)]

    return smoothed_t, smoothed_wv