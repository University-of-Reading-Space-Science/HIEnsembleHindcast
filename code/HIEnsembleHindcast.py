# Standard library
from astropy.time import Time
import astropy.units as u
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
import os
import pandas as pd
import sunpy.coordinates.sun as sn
import tables

# Our own library for using spice with STEREO (https://github.com/LukeBarnard/stereo_spice)
from stereo_spice.coordinates import StereoSpice

# Local packages
import HI_analysis as hip
import HUXt as H

spice = StereoSpice()


def load_swpc_cme_forecasts(event):
    """
    Load in a dictionary of SWPC CME forecast parameters for each solar stormwatch event.
    Parameters
    ----------
    event: A string key to identify which event to load. Should be ssw_007, ssw_008, ssw_009, or ssw_012.

    Returns
    -------
    cme: A dictionary containing the parameters of each event.

    """
    
    event_list = ['ssw_007', 'ssw_008', 'ssw_009', 'ssw_012']
    if event not in event_list:
        raise ValueError("Warning: {} not in event list.".format(event))
        
    kms = u.km / u.s
    
    if event == 'ssw_007':
        cme = {'t_obs': Time('2012-08-31T22:46:00'), 'r_obs': 21.5*u.solRad, 'width': 66*u.deg, 'lon': -30*u.deg,
               'lat': 0*u.deg, 'v': 1010*kms, 't_arr_pred': Time('2012-09-03T17:00:00'),
               't_arr_obs': Time('2012-09-03T11:23:00')}
    
    elif event == 'ssw_008':
        cme = {'t_obs': Time('2012-09-28T03:49:00'), 'r_obs': 21.5*u.solRad, 'width': 110*u.deg, 'lon': 20*u.deg,
               'lat': 4*u.deg, 'v': 872*kms, 't_arr_pred': Time('2012-09-30T15:00:00'),
               't_arr_obs': Time('2012-09-30T22:13:00')}
    
    elif event == 'ssw_009':
        cme = {'t_obs': Time('2012-10-05T08:47:00'), 'r_obs': 21.5*u.solRad, 'width': 84*u.deg, 'lon': 9*u.deg,
               'lat': -24*u.deg, 'v': 698*kms, 't_arr_pred': Time('2012-10-08T15:00:00'),
               't_arr_obs': Time('2012-10-08T04:31:00')}
    
    elif event == 'ssw_012':
        cme = {'t_obs': Time('2012-11-20T17:40:00'), 'r_obs': 21.5*u.solRad, 'width': 94*u.deg, 'lon': 22*u.deg,
               'lat': 20*u.deg, 'v': 664*kms, 't_arr_pred': Time('2012-11-23T17:00:00'),
               't_arr_obs': Time('2012-11-23T21:12:00')}
        
    return cme


def run_huxt_ensemble(ssw_event, n_ensemble=100):
    """
    Produce a determinisitc and ensemble of HUXt runs for a specified solar stormwatch event. For the deterministic run,
    both the full model solution, and the CME profile are saved in data>HUXt. For the ensemble, only the CME profiles
    are saved in data>HUXt, as this reduces the storage requirements significantly.
    Parameters
    ----------
    ssw_event: String identifying which event to run the ensemble for. Should be ssw_007, ssw_008, ssw_009, or ssw_012.
    n_ensemble: Number of ensemble members to include, defaults to 200.
    Returns
    -------
    A set of files in data>HUXt for the specified event.
    """
    
    # Add a check n_ensemble so that is integer and sensible
    
    swpc_cme = load_swpc_cme_forecasts(ssw_event)
    
    # Get the carrington rotation number, and Earth's coordinates, at SWPCs initial observation time.
    cr_num = np.fix(sn.carrington_rotation_number(swpc_cme['t_obs']))
    ert = H.Observer('EARTH', swpc_cme['t_obs'])

    print("Carrington rotation: {}".format(cr_num))
    print("Earth Carrington Lon at init: {:3.2f}".format(ert.lon_c.to(u.deg)))
    print("Earth HEEQ Lat at init: {:3.2f}".format(ert.lat.to(u.deg)))
    
    # Set up HUXt for a 5 day simulation
    vr_in, br_in = H.Hin.get_MAS_long_profile(cr_num, ert.lat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                   br_boundary=br_in, simtime=5*u.day, dt_scale=4)

    # Deterministic run first:
    # Get time of CME at inner boundary, assuming fixed speed.
    r_ib = model.r.min()
    dt = ((r_ib - swpc_cme['r_obs']) / swpc_cme['v']).to('s')
    thickness = 5 * u.solRad
    # Setup a ConeCME with these parameters
    conecme = H.ConeCME(t_launch=dt, longitude=swpc_cme['lon'], latitude=swpc_cme['lat'],
                        width=swpc_cme['width'], v=swpc_cme['v'], thickness=thickness)

    # Run HUXt with this ConeCME
    tag = "{}_{}".format(ssw_event, 'deterministic')
    model.solve([conecme], save=True, save_cmes=True, tag=tag)
    
    # Now produce ensemble of HUXt runs with perturbed ConeCME parameters
    np.random.seed(1987)

    lon_spread = 20 * u.deg
    lat_spread = 20 * u.deg
    width_spread = 20 * u.deg
    v_spread = 200 * model.kms
    thickness_spread = 2 * u.solRad
    r_init_spread = 3 * u.solRad

    for i in range(n_ensemble):

        lon = swpc_cme['lon'] + np.random.uniform(-1, 1, 1)[0] * lon_spread
        lat = swpc_cme['lat'] + np.random.uniform(-1, 1, 1)[0] * lat_spread
        width = swpc_cme['width'] + np.random.uniform(-1, 1, 1)[0] * width_spread
        v = swpc_cme['v'] + np.random.uniform(-1, 1, 1)[0] * v_spread
        thickness = 5.0*u.solRad + np.random.uniform(-1, 1, 1)[0] * thickness_spread

        # Workout time of CME at inner boundary, assuming fixed speed.
        r_init = swpc_cme['r_obs'] + np.random.uniform(-1, 1, 1)[0] * r_init_spread
        r_ib = model.r.min()
        dt = ((r_ib - r_init) / v).to('s')

        # Setup the ConeCME and run the model.
        conecme = H.ConeCME(t_launch=dt, longitude=lon, latitude=lat, width=width, v=v, thickness=thickness)
        tag = "{}_ensemble_{:02d}".format(ssw_event, i)
        model.solve([conecme], save_cmes=True, tag=tag)
    
    return


def track_cme_flanks(ssw_event, fast=True):
    """
    Compute the CME flank elongation for each ensemble member and save to file.
    Parameters
    ----------
    ssw_event: String giving the name of the SWPC event to analyse
    fast: Boolean, default True, of whether to use a faster version of the flank tracking algorithm. Saves a
          significant amount of time, and works for the events studied here. Might not generalise well to other events.
    Returns
    -------
    Files in data>out_data, with name format ssw_event_ensemble_sta.csv and ssw_event_ensemble_stb.csv

    """
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "CME_*{}*_ensemble_*.hdf5".format(ssw_event))
    ensemble_files = glob.glob(path)
    n_ens = len(ensemble_files)

    # Produce a dictionary of keys of column headings for the dataframes 
    # storing the ensemble of time elonation profiles
    keys = []
    parameters = ['el', 'pa', 'r', 'lon']
    for param in parameters:
        for i in range(n_ens):
            keys.append("{}_{:02d}".format(param, i))

    keys = {k: 0 for k in keys}

    # Loop over the ensemble files, pull out the elongation profiles and compute arrival time.
    for i, file in enumerate(ensemble_files):

        cme_list = H.load_cme_file(file)
        cme = cme_list[0] 

        # Compute the time-elongation profiles of the CME flanks from STA and STB,
        # and store into dataframes for each set of ensembles
        
        if fast:
            hxta, hxtb = huxt_t_e_profile_fast(cme)
        else:
            hxta, hxtb = huxt_t_e_profile(cme)
            
        if i == 0:    
            # Make pandas array to store all ensemble t_e_profiles.
            keys['time'] = hxta['time']
            ensemble_sta = pd.DataFrame(keys)
            ensemble_stb = pd.DataFrame(keys)

        # Update the ensemble dataframes
        for key in ['r', 'lon', 'el', 'pa']:
            e_key = "{}_{:02d}".format(key, i)
            ensemble_sta[e_key] = hxta[key]
            ensemble_stb[e_key] = hxtb[key]

    out_path = project_dirs['out_data']
    out_name = ssw_event + '_ensemble_sta.csv'
    ensemble_sta.to_csv(os.path.join(out_path, out_name))
    out_name = ssw_event + '_ensemble_stb.csv'
    ensemble_stb.to_csv(os.path.join(out_path, out_name))
    return


def compute_ssw_profile(ssw_event):
    """
    Compute the Solar Stormwatch time-elongation profile along the position angle corresponding to the model solution.

    Parameters
    ----------
    ssw_event: String giving the name of the SWPC event to analyse

    Returns
    -------
    Files in data>out_data which have the name format ssw_event_sta.csv and ssw_event_stb.csv.

    """
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "CME_*{}*_ensemble_*.hdf5".format(ssw_event))
    ensemble_files = glob.glob(path)
    n_ens = len(ensemble_files)
    
    # Load in the HUXt ensemble flanks
    sta_flanks = ssw_event + '_ensemble_sta.csv'
    ensemble_sta = pd.read_csv(os.path.join(project_dirs['out_data'], sta_flanks))
    stb_flanks = ssw_event + '_ensemble_stb.csv'
    ensemble_stb = pd.read_csv(os.path.join(project_dirs['out_data'], stb_flanks))

    # Compute ensemble average of average PA of the HUXt profiles in the HI FOV.
    # PA variability and trend is small enough to assume constant.
    el_keys = ["el_{:02d}".format(i) for i in range(n_ens)]
    pa_keys = ["pa_{:02d}".format(i) for i in range(n_ens)]

    sta_pa_lst = []
    stb_pa_lst = []
    for i, (ek, pk) in enumerate(zip(el_keys, pa_keys)):
        id_hi = ensemble_sta[ek] <= 25.0   
        if not np.isnan(ensemble_stb.loc[id_hi, pk].mean()):
            sta_pa_lst.append(ensemble_sta.loc[id_hi, pk].mean())
            
        id_hi = ensemble_stb[ek] <= 25.0
        if not np.isnan(ensemble_stb.loc[id_hi, pk].mean()):
            stb_pa_lst.append(ensemble_stb.loc[id_hi, pk].mean())
            
    print("****************")
    print("HI1A Flank profiles:{}".format(len(sta_pa_lst)))
    print("HI1B Flank profiles:{}".format(len(stb_pa_lst)))
    
    sta_pa_lst = np.array(sta_pa_lst)
    sta_pa_avg = np.mean(sta_pa_lst)
    stb_pa_lst = np.array(stb_pa_lst)
    stb_pa_avg = np.mean(stb_pa_lst)

    ssw_sta = get_ssw_profile(ssw_event, 'sta', 'diff', sta_pa_avg, pa_wid=2.0)
    ssw_stb = get_ssw_profile(ssw_event, 'stb', 'diff', stb_pa_avg, pa_wid=2.0)
    out_path = project_dirs['out_data']
    sta_out_name = ssw_event + "_sta.csv"
    ssw_sta.to_csv(os.path.join(out_path, sta_out_name))
    stb_out_name = ssw_event + "_stb.csv"
    ssw_stb.to_csv(os.path.join(out_path, stb_out_name))
    return
  
    
def compute_ensemble_metrics(ssw_event):
    """
    For each ensemble member, compute the arrival at Earth, and the RMSE and weighting based on comparison with the
    solar stormwatch profile of this event. Save these metrics to file.

    Parameters
    ----------
    ssw_event: String giving the name of the SWPC event to analyse

    Returns
    -------
    A file in data>out_data, with name format ssw_event_ensemble_metrics.csv

    """
    # List the ensemble files, and set up space for results of comparisons
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "CME_*{}*_ensemble_*.hdf5".format(ssw_event))
    ensemble_files = glob.glob(path)
    n_ens = len(ensemble_files)
    metrics = pd.DataFrame({'arrival_time': np.zeros(n_ens), 'transit_time': np.zeros(n_ens),
                            'rmse_a': np.zeros(n_ens), 'rmse_b': np.zeros(n_ens), 'hit': np.zeros(n_ens)})
    
    # Load in the HUXt ensemble flanks
    sta_flanks = ssw_event + '_ensemble_sta.csv'
    ensemble_sta = pd.read_csv(os.path.join(project_dirs['out_data'], sta_flanks))
    stb_flanks = ssw_event + '_ensemble_stb.csv'
    ensemble_stb = pd.read_csv(os.path.join(project_dirs['out_data'], stb_flanks))
    # Load in the SSW profiles
    ssw_sta_name = ssw_event + "_sta.csv"
    ssw_sta = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_sta_name))
    ssw_stb_name = ssw_event + "_stb.csv"
    ssw_stb = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_stb_name))
    
    # Problem with ssw_012, first two elongation points are erroneous. 
    # Exclude from the analysis.
    if ssw_event == 'ssw_012':
        fig, ax = plt.subplots()
        ax.plot(ssw_stb['time'], ssw_stb['el'], 'ko', label='All SSW HI1B')
        ssw_stb = ssw_stb.loc[2:, :]
        ax.plot(ssw_stb['time'], ssw_stb['el'], 'r.', label='Outlier excluded data')
        path = os.path.join(project_dirs['HUXt_figures'], "ssw_012_outlier_exclude_check.png")
        fig.savefig(path)
        plt.close(fig)

    # Loop over the ensemble files, pull out the elongation profiles and compute arrival time.
    for i, file in enumerate(ensemble_files):

        cme_list = H.load_cme_file(file)
        cme = cme_list[0] 

        # Compute the arrival time for this event    
        if not np.isnan(cme.earth_transit_time.value):
            metrics.loc[i, 'arrival_time'] = cme.earth_arrival_time.datetime
            metrics.loc[i, 'transit_time'] = cme.earth_transit_time.value
            metrics.loc[i, 'hit'] = 1
        else:
            metrics.loc[i, 'arrival_time'] = pd.NaT
            metrics.loc[i, 'transit_time'] = pd.NaT

    # Make arrival time a datetime, as currently object
    metrics['arrival_time'] = pd.to_datetime(metrics['arrival_time'])

    # Now use the SSW profiles to compute the rms difference with each huxt ensemble, and update ensemble_metrics.
    # Get keys of elon profiles 
    el_keys = []
    for col in ensemble_sta:
        if col[0:2] == 'el':
            el_keys.append(col)
    
    for i, ek in enumerate(el_keys):

        hxt_sta = pd.DataFrame({'time': ensemble_sta['time'], 'el': ensemble_sta[ek]})
        rms, n_rms_samp = compute_rmse(hxt_sta, ssw_sta)
        metrics.loc[i, 'rmse_a'] = rms

        hxt_stb = pd.DataFrame({'time': ensemble_stb['time'], 'el': ensemble_stb[ek]})
        rms, n_rms_samp = compute_rmse(hxt_stb, ssw_stb)
        metrics.loc[i, 'rmse_b'] = rms

    # Calculate normalised weights. Normalise after computing average weight,
    # as otherwise systematic difference between A and B is removed
    metrics['w_a'] = 1.0 / metrics['rmse_a']
    metrics['w_b'] = 1.0 / metrics['rmse_b']
    metrics['w_avg'] = metrics.loc[:, ['w_a', 'w_b']].mean(axis=1)

    metrics['w_a'] = metrics['w_a'] / metrics['w_a'].sum()
    metrics['w_b'] = metrics['w_b'] / metrics['w_b'].sum()
    metrics['w_avg'] = metrics['w_avg'] / metrics['w_avg'].sum()

    out_path = project_dirs['out_data']
    out_name = ssw_event + "_ensemble_metrics.csv"
    metrics.to_csv(os.path.join(out_path, out_name))
    return


def huxt_t_e_profile(cme):
    """
    Compute the time elongation profile of the flank of a ConeCME in HUXt, from both the STEREO-A or STEREO-B
    persepctive. Uses stereo_spice in a loop, which can be quite slow.
    Parameters
    ----------
    cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has been populated).

    Returns
    -------
    sta_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STA's perspective, including the
                 time, elongation, position angle, and HEEQ radius and longitude.
    stb_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STB's perspective, including the
                 time, elongation, position angle, and HEEQ radius and longitude.
    """
    times = Time([coord['time'] for i, coord in cme.coords.items()])

    sta_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon', 'pa'])
    stb_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon', 'pa'])

    sta_profile['time'] = times.jd
    stb_profile['time'] = times.jd

    for i, coord in cme.coords.items():

        if len(coord['r']) == 0:
            continue

        lon_cme = coord['lon']
        r_cme = coord['r']

        e_sta = np.zeros(r_cme.shape)
        pa_sta = np.zeros(r_cme.shape)

        e_stb = np.zeros(r_cme.shape)
        pa_stb = np.zeros(r_cme.shape)

        for j, (r, l) in enumerate(zip(r_cme, lon_cme)):
            coords = np.array([r.to('km').value, l.to('rad').value, 0])
            # STA first
            coord_hpr = spice.convert_lonlat(times[i], coords, 'HEEQ', 'HPR', observe_dst='STA', degrees=False)
            e_sta[j] = coord_hpr[1]
            pa_sta[j] = coord_hpr[2]

            # STB
            coord_hpr = spice.convert_lonlat(times[i], coords, 'HEEQ', 'HPR', observe_dst='STB', degrees=False)
            e_stb[j] = coord_hpr[1]
            pa_stb[j] = coord_hpr[2]

        id_sta_fov = (pa_sta >= 0) & (pa_sta < np.pi)
        if np.any(id_sta_fov):
            lon_sub = lon_cme[id_sta_fov]
            r_sub = r_cme[id_sta_fov]
            e_sub = e_sta[id_sta_fov]
            p_sub = pa_sta[id_sta_fov]
            id_flank = np.argmax(e_sub)
            sta_profile.loc[i, 'lon'] = lon_sub[id_flank].value
            sta_profile.loc[i, 'r'] = r_sub[id_flank].value
            sta_profile.loc[i, 'el'] = np.rad2deg(e_sub[id_flank])
            sta_profile.loc[i, 'pa'] = np.rad2deg(p_sub[id_flank])
        else:
            sta_profile.loc[i, 'lon'] = np.NaN
            sta_profile.loc[i, 'r'] = np.NaN
            sta_profile.loc[i, 'el'] = np.NaN
            sta_profile.loc[i, 'pa'] = np.NaN

        id_stb_fov = (pa_stb >= np.pi) & (pa_stb < 2*np.pi)
        if np.any(id_stb_fov):
            lon_sub = lon_cme[id_stb_fov]
            r_sub = r_cme[id_stb_fov]
            e_sub = e_stb[id_stb_fov]
            p_sub = pa_stb[id_stb_fov]
            id_flank = np.argmax(e_sub)
            stb_profile.loc[i, 'lon'] = lon_sub[id_flank].value
            stb_profile.loc[i, 'r'] = r_sub[id_flank].value
            stb_profile.loc[i, 'el'] = np.rad2deg(e_sub[id_flank])
            stb_profile.loc[i, 'pa'] = np.rad2deg(p_sub[id_flank])
        else:
            stb_profile.loc[i, 'lon'] = np.NaN
            stb_profile.loc[i, 'r'] = np.NaN
            stb_profile.loc[i, 'el'] = np.NaN
            stb_profile.loc[i, 'pa'] = np.NaN
            
    keys = ['lon', 'r', 'el', 'pa']
    sta_profile[keys] = sta_profile[keys].astype(np.float64)
    
    keys = ['lon', 'r', 'el', 'pa']
    stb_profile[keys] = stb_profile[keys].astype(np.float64)
            
    return sta_profile, stb_profile


def huxt_t_e_profile_fast(cme):
    """
    Compute the time elongation profile of the flank of a ConeCME in HUXt, from both the STEREO-A or STEREO-B
    perspective. A faster, but less reliable, version of computing the CME flank with huxt_t_e_profile. Rather than
    using stereo_spice for the full calculation, which is a bit slow, this function does it's own calculation of the
    flank elongation, and then uses stereo_spice to compute the flank position angle. This might fail for some
    geometries where the elongation is technically larger along PA angles not visible to either STA or STB. However,
    this agrees with huxt_t_e_profile for the deterministic runs, so I think is safe for the events in this study.

    Parameters
    ----------
    cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has been populated).
    Returns
    -------
    sta_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STA's perspective, including the
                time, elongation, position angle, and HEEQ radius and longitude.
    stb_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STB's perspective, including the
                time, elongation, position angle, and HEEQ radius and longitude.
    """
    times = Time([coord['time'] for i, coord in cme.coords.items()])
    sta = H.Observer('STA', times)
    stb = H.Observer('STB', times)

    sta_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon', 'pa'])
    stb_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon', 'pa'])

    sta_profile['time'] = times.jd
    stb_profile['time'] = times.jd

    for i, coord in cme.coords.items():

        if len(coord['r']) == 0:
            continue

        r_sta = sta.r[i]
        x_sta = sta.r[i] * np.cos(sta.lat[i]) * np.cos(sta.lon[i])
        y_sta = sta.r[i] * np.cos(sta.lat[i]) * np.sin(sta.lon[i])
        z_sta = sta.r[i] * np.sin(sta.lat[i])

        r_stb = stb.r[i]
        x_stb = stb.r[i] * np.cos(stb.lat[i]) * np.cos(stb.lon[i])
        y_stb = stb.r[i] * np.cos(stb.lat[i]) * np.sin(stb.lon[i])
        z_stb = stb.r[i] * np.sin(stb.lat[i])

        lon_cme = coord['lon']
        lat_cme = coord['lat']
        r_cme = coord['r']

        x_cme = r_cme * np.cos(lat_cme) * np.cos(lon_cme)
        y_cme = r_cme * np.cos(lat_cme) * np.sin(lon_cme)
        z_cme = r_cme * np.sin(lat_cme)
        #############
        # Compute the observer CME distance, S, and elongation

        x_cme_s = x_cme - x_sta
        y_cme_s = y_cme - y_sta
        z_cme_s = z_cme - z_sta
        s = np.sqrt(x_cme_s**2 + y_cme_s**2 + z_cme_s**2)

        numer = (r_sta**2 + s**2 - r_cme**2).value
        denom = (2.0 * r_sta * s).value
        e_sta = np.arccos(numer / denom)

        x_cme_s = x_cme - x_stb
        y_cme_s = y_cme - y_stb
        z_cme_s = z_cme - z_stb
        s = np.sqrt(x_cme_s**2 + y_cme_s**2 + z_cme_s**2)

        numer = (r_stb**2 + s**2 - r_cme**2).value
        denom = (2.0 * r_stb * s).value
        e_stb = np.arccos(numer / denom)

        # Find the flank coordinate
        id_sta_flank = np.argmax(e_sta)
        id_stb_flank = np.argmax(e_stb)

        e_sta = e_sta[id_sta_flank]
        e_stb = e_stb[id_stb_flank]

        # STA PA first
        r = r_cme[id_sta_flank].to('km').value
        lon = lon_cme[id_sta_flank].to('rad').value
        lat = lat_cme[id_sta_flank].to('rad').value
        coords = np.array([r, lon, lat])
        coord_hpr = spice.convert_lonlat(times[i], coords, 'HEEQ', 'HPR', observe_dst='STA', degrees=False)
        e_sta_2 = coord_hpr[1]
        pa_sta = coord_hpr[2]

        # STB PA
        r = r_cme[id_stb_flank].to('km').value
        lon = lon_cme[id_stb_flank].to('rad').value
        lat = lat_cme[id_stb_flank].to('rad').value
        coords = np.array([r, lon, lat])
        coord_hpr = spice.convert_lonlat(times[i], coords, 'HEEQ', 'HPR', observe_dst='STB', degrees=False)
        e_stb_2 = coord_hpr[1]
        pa_stb = coord_hpr[2]
        
        id_sta_fov = (pa_sta >= 0) & (pa_sta < np.pi)
        if id_sta_fov & np.allclose(e_sta, e_sta_2):
            sta_profile.loc[i, 'lon'] = lon_cme[id_sta_flank].value
            sta_profile.loc[i, 'r'] = r_cme[id_sta_flank].value
            sta_profile.loc[i, 'el'] = np.rad2deg(e_sta_2)
            sta_profile.loc[i, 'pa'] = np.rad2deg(pa_sta)
        else:
            sta_profile.loc[i, 'lon'] = np.NaN
            sta_profile.loc[i, 'r'] = np.NaN
            sta_profile.loc[i, 'el'] = np.NaN
            sta_profile.loc[i, 'pa'] = np.NaN

        id_stb_fov = (pa_stb >= np.pi) & (pa_stb < 2*np.pi)
        if id_stb_fov & np.allclose(e_stb, e_stb_2):
            stb_profile.loc[i, 'lon'] = lon_cme[id_stb_flank].value
            stb_profile.loc[i, 'r'] = r_cme[id_stb_flank].value
            stb_profile.loc[i, 'el'] = np.rad2deg(e_stb_2)
            stb_profile.loc[i, 'pa'] = np.rad2deg(pa_stb)
        else:
            stb_profile.loc[i, 'lon'] = np.NaN
            stb_profile.loc[i, 'r'] = np.NaN
            stb_profile.loc[i, 'el'] = np.NaN
            stb_profile.loc[i, 'pa'] = np.NaN
            
    keys = ['lon', 'r', 'el', 'pa']
    sta_profile[keys] = sta_profile[keys].astype(np.float64)

    keys = ['lon', 'r', 'el', 'pa']
    stb_profile[keys] = stb_profile[keys].astype(np.float64)

    return sta_profile, stb_profile


def compute_rmse(hxt, ssw):
    """
    Compute the root mean square error between the HUXt and ssw time elongation profiles of the CME flank. The HUXt
    profiles is interpolated onto the solar stormwatch profile to compute the comparison.
    Parameters
    ----------
    hxt: Pandas dataframe containing the ConeCME profile from HUXt
    ssw: Pandas dataframe containing the observed CME profile from either STEREO-A or STEREO-B
    Returns
    -------
    rms: The root mean square error between these profiles, ignoring NaNs
    n_rms_samp: The number of valid points in the profile comparisons.
    """
    elon_interp = np.interp(ssw['time'].values, hxt['time'].values, hxt['el'].values, left=np.NaN, right=np.NaN)
    hxt_interp = pd.DataFrame({'time': ssw['time'].values, 'el': elon_interp})
    de = (hxt_interp['el'] - ssw['el'])**2
    n_rms_samp = np.sum(np.isfinite(de))
    rms = np.sqrt(de.mean(skipna=True))
    return rms, n_rms_samp


def get_ssw_profile(ssw_event, craft, img, pa_center, pa_wid=1.0):
    """
    Compute the Solar Stormwatch profile of the CME along a fixed position angle window, for either STA or STB.
    Parameters
    ----------
    ssw_event: String identifier of which SWPC event to analyse. Should be ssw_007, ssw_008, ssw_009, or ssw_012.
    craft: String identifier of which STEREO craft to analyse, should be 'STA' or 'STB'.
    img: String identifier of whether to retrieve the ssw profiles of normal, or differenced images. Should be
        'norm', or 'diff'
    pa_center: Float value of the central position angle to track the CME front along. In degrees.
    pa_wid: Float value of the width of the position angle slice to track the CME front along. Defaults to 1 degree

    Returns
    -------
    profile: A pandas dataframe giving the consensus time-elongation profile of the observed CME front derived from the
            full distribution of solar stormwatch classifications.

    """
    # Open up the SSW data
    project_dirs = H._setup_dirs_()
    ssw_out = tables.open_file(project_dirs['SSW_data'], mode="r")
    # Pull out event
    ssw_path = "/".join(['', ssw_event, craft, img])
    event = ssw_out.get_node(ssw_path)
    
    # Now for each time, look up elongation at position angle nearest to requested pa.
    times = []
    el_best = []
    el_lo = []
    el_hi = []
    for cme_slice in event:
        frame_time = Time(cme_slice._v_title, format='isot', scale='utc')
        # Stash time
        times.append(frame_time.datetime)
        # Get the CME front data
        cme = pd.DataFrame.from_records(cme_slice.cme_coords.read())
        cme.replace(to_replace=[99999], value=np.NaN, inplace=True)
        # Look up the indices of this position angle slice, and average the elongation coords in this
        # window.
        id_pa = (cme['pa'] >= (pa_center - pa_wid)) & (cme['pa'] <= (pa_center + pa_wid))
        el_best.append(np.nanmean(cme['el'][id_pa]))
        el_lo.append(np.nanmean(cme['el_lo'][id_pa]))
        el_hi.append(np.nanmean(cme['el_hi'][id_pa]))

    # Make dataframe of the raw ssw profile
    # Get array of PA's to pass out too
    pa = np.zeros(len(times)) + pa_center
    # Convert profile into a dataframe and return.
    profile = pd.DataFrame({'time': times, 'el': el_best, 'el_lo': el_lo, 'el_hi': el_hi, 'pa': pa})

    profile['el_dlo'] = profile['el'] - profile['el_lo']
    profile['el_dhi'] = profile['el_hi'] - profile['el']
    # Set the time error to zero, as well defined for SSW
    profile['time_err'] = profile['time'] - profile['time']
    # Add in julian dates
    profile['time'] = pd.DatetimeIndex(profile['time']).to_julian_date()

    ssw_out.close()
    return profile


def plot_huxt_and_hi_schematic(ssw_event, time):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.

    Parameters
    ----------
    ssw_event: String identifier of which SWPC event to analyse.
    time: Time to look up closet model time to (with an astropy.unit of time).

    Returns
    -------
    fig: Figure handle.
    ax: Axes handle.
    """

    # Load in the deterministic HUXt solution, SSW data, and HI images.
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "HUXt_*{}*_deterministic.hdf5".format(ssw_event))
    deterministic_run = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(deterministic_run)
    cme = cme_list[0]

    hxt_sta, hxt_stb = huxt_t_e_profile_fast(cme)
    
    # LOAD IN THE CORRECT PA FROM THE SSW DATA FOR PLOTTING
    ssw_sta_name = ssw_event + "_sta.csv"
    ssw_sta_hxt = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_sta_name))
    ssw_stb_name = ssw_event + "_stb.csv"
    ssw_stb_hxt = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_stb_name))   
    
    sta_hxt_pa = ssw_sta_hxt.loc[0, 'pa']
    stb_hxt_pa = ssw_stb_hxt.loc[0, 'pa']
    print("{} STA PA of HUXt: {:3.2f}".format(ssw_event, sta_hxt_pa))
    print("{} STB PA of HUXt: {:3.2f}".format(ssw_event, stb_hxt_pa))
    del ssw_sta_hxt
    del ssw_stb_hxt
    
    id_time_out = np.argmin(np.abs(model.time_out - time))
    time_out = model.time_init + model.time_out[id_time_out]

    times = model.time_init + model.time_out
    sta = H.Observer('STA', times)
    stb = H.Observer('STB', times)

    # Load in the HI images
    hi1a_fc, hi1a_fp = find_hi_files(ssw_event, 'sta', time_out)
    hi1a_map = hip.get_image_diff(hi1a_fc, hi1a_fp, align=True, smoothing=True)

    hi1b_fc, hi1b_fp = find_hi_files(ssw_event, 'stb', time_out)
    hi1b_map = hip.get_image_diff(hi1b_fc, hi1b_fp, align=True, smoothing=True)

    # Load the SSW classifications for the HI images.
    img = 'diff'
    # Open up the SSW data
    ssw_out = tables.open_file(project_dirs['SSW_data'], mode="r")
    plot_hi1a_flag = True
    plot_hi1b_flag = True
    for craft, hi_map in zip(['sta', 'stb'], [hi1a_map, hi1b_map]):

        # Pull out event
        ssw_path = "/".join(['', ssw_event, craft, img])
        event = ssw_out.get_node(ssw_path)

        key = hi_map.date.strftime('T%Y%m%d_%H%M')+'01'
        
        # There is a bug in the naming/timing of the files for STB. This is a workaround.
        if key == "T20121121_033001":
            key = "T20121121_032901"
          
        try:
            cme_slice = event[key]

            # Get the CME front data
            cme = pd.DataFrame.from_records(cme_slice.cme_coords.read())
            cme.replace(to_replace=[99999], value=np.NaN, inplace=True)

            if craft == 'sta':
                ssw_sta = cme.copy()
            elif craft == 'stb':
                ssw_stb = cme.copy()
        except:
            if craft == 'sta':
                ssw_sta = None
                plot_hi1a_flag = False
            elif craft == 'stb':
                ssw_stb = None
                plot_hi1b_flag = False
                
    ssw_out.close()

    fig = plt.figure(figsize=(27, 10))
    
    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon = model.lon_grid.value.copy()
    rad = model.r_grid.value.copy()
    v = model.v_grid_cme.value[id_t, :, :].copy()
    
    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap = mpl.cm.viridis
    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    dv = 10
    levels = np.arange(200, 800+dv, dv)
    
    ax = plt.subplot(131, polar=True)
    ax2 = plt.subplot(132, polar=False)
    ax3 = plt.subplot(133, polar=False)
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries
    cme = model.cmes[0]
    ax.plot(cme.coords[id_t]['lon'], cme.coords[id_t]['r'], '-', color='darkorange', linewidth=3, zorder=3)

    for body, style in zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'], ['co', 'mo', 'ko', 'rs', 'y^']):
        obs = model.get_observer(body)
        if body != 'STB':
            ax.plot(obs.lon[id_t], obs.r[id_t], style, markersize=16, label=body)
        elif body == 'STB':
            ax.plot(obs.lon[id_t], obs.r[id_t], '^', color='fuchsia', markersize=16, label=body)
        
    #####################################################
    # Add on HI1A FOV, 4 to 25 
    rsa = np.mean(sta.r)
    lsa = np.mean(sta.lon)
    xsa = rsa * np.cos(lsa)
    ysa = rsa * np.sin(lsa)

    rsb = np.mean(stb.r)
    lsb = np.mean(stb.lon)
    xsb = rsb * np.cos(lsb)
    ysb = rsb * np.sin(lsb)
    
    sta_patch = [[lsa.value, rsa.value]]
    stb_patch = [[lsb.value, rsb.value]]

    # Get plane of sky coord of elon lims
    for el in [4.0, 24.0]:
        # STA
        rp = rsa * np.tan(el*u.deg)
        lp = lsa - 90*u.deg

        xp = rp * np.cos(lp)
        yp = rp * np.sin(lp)
        grad = (yp - ysa) / (xp - xsa)
        c = ysa - grad*xsa
        xf = 250 * u.solRad
        yf = grad*xf + c
        rf = np.sqrt(xf**2 + yf**2)
        lf = np.arctan2(yf, xf)
        sta_patch.append([lf.value, rf.value])

        # STB
        rp = rsb * np.tan(el*u.deg)
        lp = lsb + 90*u.deg

        xp = rp * np.cos(lp)
        yp = rp * np.sin(lp)
        grad = (yp - ysb) / (xp - xsb)
        c = ysb - grad*xsb
        xf = 250 * u.solRad
        yf = grad*xf + c
        rf = np.sqrt(xf**2 + yf**2)
        lf = np.arctan2(yf, xf)
        stb_patch.append([lf.value, rf.value])
        
    sta_patch = mpl.patches.Polygon(np.array(sta_patch), color='r', alpha=0.3, zorder=1)
    ax.add_patch(sta_patch)
    stb_patch = mpl.patches.Polygon(np.array(stb_patch), color='fuchsia', alpha=0.3, zorder=1)
    ax.add_patch(stb_patch)

    # Add on the flanks.
    id_t = np.argmin(np.abs(model.time_out - time))
    ax.plot([lsa.value, hxt_sta.loc[id_t, 'lon']], [rsa.value, hxt_sta.loc[id_t, 'r']], 'r-', linewidth=2)
    ax.plot(hxt_sta.loc[id_t, 'lon'], hxt_sta.loc[id_t, 'r'], 'rX', markersize=15, zorder=4)

    ax.plot([lsb.value, hxt_stb.loc[id_t, 'lon']], [rsb.value, hxt_stb.loc[id_t, 'r']], '-', color='fuchsia', linewidth=2)
    ax.plot(hxt_stb.loc[id_t, 'lon'], hxt_stb.loc[id_t, 'r'], 'd', color='fuchsia', markersize=15, zorder=4)
    ####################################################

    # Add on a legend.
    fig.legend(ncol=5, loc='lower left', bbox_to_anchor=(0.025, 0.0), frameon=False, handletextpad=0.2, columnspacing=1)

    ax.set_ylim(0, 240)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_facecolor('slategrey')
    fig.subplots_adjust(left=0.0, bottom=0.15, right=0.99, top=0.99, wspace=-0.12)
    
    # Add color bar
    pos = ax.get_position()
    dw = 0.005
    dh = 0.045
    left = pos.x0 + dw
    bottom = pos.y0 - dh
    wid = pos.width - 2*dw
    cbaxes = fig.add_axes([left, bottom, wid, 0.03])
    cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
    cbar1.set_label("Solar Wind Speed (km/s)")
    cbar1.set_ticks(np.arange(200, 900, 100))

    # Add label
    label = (model.time_init + model.time_out[id_t]).strftime("%Y-%m-%dT%H:%M")
    fig.text(0.25, pos.y0, label, fontsize=16)
    label = "HUXt2D"
    fig.text(0.07, pos.y0, label, fontsize=16)
    
    ###################################################
    normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
    
    for a, hi_map, ssw, plot_flag in zip([ax2, ax3], [hi1a_map, hi1b_map], [ssw_sta, ssw_stb],
                                         [plot_hi1a_flag, plot_hi1b_flag]):
        
        img = mpl.cm.gray(normalise(hi_map.data), bytes=True)

        # Plot out the raw frame
        if hi_map.observatory == "STEREO A":
            color = 'r'
            pa = sta_hxt_pa
            fmt = 'X'
        elif hi_map.observatory == "STEREO B":
            color = 'fuchsia'
            pa = stb_hxt_pa
            fmt = 'd'

        a.imshow(img, origin='lower')
        
        el_arr = np.arange(0,25,1)
        for dpa in [-2.0, 2.0]:
            pa_arr = np.zeros(el_arr.shape) + pa + dpa
            x, y = hip.convert_hpr_to_pix(el_arr*u.deg, pa_arr*u.deg, hi_map)
            a.plot(x, y, ':', color='lawngreen', linewidth=3)

        if plot_flag:
            a.plot(ssw['x'], ssw['y'], '-', color=color, linewidth=3 )
            a.plot(ssw['x_lo'], ssw['y_lo'], '--', color=color, linewidth=3)
            a.plot(ssw['x_hi'], ssw['y_hi'], '--', color=color, linewidth=3)
            id_pa = np.argmin(np.abs(ssw['pa'] - pa))
            a.plot(ssw.loc[id_pa, 'x'], ssw.loc[id_pa, 'y'], fmt, color=color, markersize=15)

        a.set_xlim(1, 1023)
        a.set_ylim(1, 1023)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    box_col = 'k'
    txt_x = 0.008
    txt_y = 0.96
    txt_fs = 18
    box = {'facecolor': box_col}
    label = "HI1A {}".format(hi1a_map.date.strftime("%Y-%m-%dT%H:%M"))
    ax2.text(txt_x, txt_y, label, transform=ax2.transAxes, fontsize=txt_fs, color='w', bbox=box)
    label = "HI1B {}".format(hi1b_map.date.strftime("%Y-%m-%dT%H:%M"))
    ax3.text(txt_x, txt_y, label, transform=ax3.transAxes, fontsize=txt_fs, color='w', bbox=box)
    
    project_dirs = H._setup_dirs_()
    fig_name = os.path.join(project_dirs['HUXt_figures'], 'figure_1_huxt_and_hi_{}.png'.format(ssw_event))
    fig.savefig(fig_name)
    plt.close(fig)
    return 


def find_hi_files(ssw_event, craft, time_out):
    """
    Find the locally stored HI1 data at the time closest to time_out.
    Parameters
    ----------
    ssw_event: String identifier of which SWPC event is being analysed.
    craft: String identifier of whether to look for STEREO-A or STEREO-B data, should be 'STA' or 'STB'
    time_out: An astropy.Time object giving the specified output time.

    Returns
    -------
    f_current: The path to the closest matching HI1 data file.
    f_previous: The path to the closest HI1 data file before f_current (for forming differenced images).
    """
    
    project_dirs = H._setup_dirs_()
    hi1_path = os.path.join(project_dirs['STEREO_HI_data'], ssw_event, craft, "*.fts")
    hi1_files = glob.glob(hi1_path)
    # Find HI1A files closest to time.
    hi1_times = []
    for f in hi1_files:
        base = os.path.basename(f)
        time_stamp = base[0:15]
        t = pd.datetime.strptime(time_stamp, "%Y%m%d_%H%M%S")
        hi1_times.append(t)
    
    hi1_times = np.array(hi1_times)
    to = time_out.to_datetime()
    if (to > hi1_times.min()) & (to <= hi1_times.max()):
        id_hi1 = np.argmin(np.abs(hi1_times - time_out.to_datetime()))

        if id_hi1 >= 1:
            f_current = hi1_files[id_hi1]
            f_previous = hi1_files[id_hi1-1]
        else:
            print('No f_previous available as time_out before available HI data span')
            f_current = hi1_files[id_hi1]
            f_previous = None
    else:
        f_current = None
        f_previous = None
    
    return f_current, f_previous


def plot_ensemble_elongation_profiles(ssw_event):
    """
    Produce a figure of the time elongation profiles of each ensemble member and the observed SSW profile.
    Parameters
    ----------
    ssw_event: String identifier of which SWPC event to analyse. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    Returns
    -------
    A figure file in figures directory with name format figure_2_time_elon_profiles_ssw_event.png
    """
    
    # List the ensemle files, and set up space for results of comparisons
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "HUXt_*{}*_deterministic.hdf5".format(ssw_event))
    deterministic_run = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(deterministic_run)
    cme = cme_list[0]

    hxta_det, hxtb_det = huxt_t_e_profile_fast(cme)
    
    # Load in the HUXt ensemble flanks
    sta_flanks = ssw_event + '_ensemble_sta.csv'
    ensemble_sta = pd.read_csv(os.path.join(project_dirs['out_data'], sta_flanks))
    stb_flanks = ssw_event + '_ensemble_stb.csv'
    ensemble_stb = pd.read_csv(os.path.join(project_dirs['out_data'], stb_flanks))
    # Load in the SSW profiles
    ssw_sta_name = ssw_event + "_sta.csv"
    ssw_sta = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_sta_name))
    ssw_stb_name = ssw_event + "_stb.csv"
    ssw_stb = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_stb_name)) 
    
    # Load in the metrics to find hits and misses
    project_dirs = H._setup_dirs_()    
    metrics = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_event + "_ensemble_metrics.csv"))
    metrics['arrival_time'] = pd.to_datetime(metrics['arrival_time'])

    # Remove misses and reindex
    id_hit = np.argwhere(metrics['hit'] == 1)
    id_miss = np.argwhere(metrics['hit'] == 0)
    
    # Set up the figure
    w = 10
    h = w / 2.0
    fig, ax = plt.subplots(1, 2, figsize=(w, h))
    
    el_keys = []
    for col in ensemble_sta:
        if col[0:2] == 'el':
            el_keys.append(col)

    for i, k in enumerate(el_keys):
        ta = ensemble_sta['time'] - model.time_init.jd
        tb = ensemble_stb['time'] - model.time_init.jd
        
        if i in id_hit:
            color = 'lightgrey'
        elif i in id_miss:
            color = 'moccasin'
        if i == 0:
            ax[0].plot(ta, ensemble_sta[k], '-', color=color, zorder=1, label='HUXt Ensemble')
            ax[1].plot(tb, ensemble_stb[k], '-', color=color, zorder=1, label='HUXt Ensemble')
        else:
            ax[0].plot(ta, ensemble_sta[k], '-', color=color, zorder=1)
            ax[1].plot(tb, ensemble_stb[k], '-', color=color, zorder=1)

    ta = hxta_det['time'] - model.time_init.jd
    ax[0].plot(ta, hxta_det['el'], '-', color='k', zorder=1, label='HUXt Deterministic')

    tb = hxtb_det['time'] - model.time_init.jd
    ax[1].plot(tb, hxtb_det['el'], '-', color='k', zorder=1, label='HUXt Deterministic')

    t = ssw_sta['time'] - model.time_init.jd
    ax[0].errorbar(t, ssw_sta['el'], yerr=[ssw_sta['el_dlo'], ssw_sta['el_dhi']], fmt='.', color='mediumblue',
                   zorder=2, label='HI1A SSW')

    t = ssw_stb['time'] - model.time_init.jd
    ax[1].errorbar(t, ssw_stb['el'], yerr=[ssw_stb['el_dlo'], ssw_stb['el_dhi']], fmt='.', color='darkviolet',
                   zorder=2, label='HI1B SSW')

    for a in ax:
        a.set_xlim(0, 1.9)
        a.set_ylim(4, 25)

        a.set_xlabel('Time after CME onset (days)')
        a.legend(loc='lower right', handletextpad=0.5, handlelength=1, columnspacing=0.5, fontsize=14, framealpha=1.0)

    ax[0].set_ylabel('Elongation (degrees)')
    ax[1].set_yticklabels([])

    fig.subplots_adjust(left=0.1, bottom=0.11, right=0.99, top=0.98, wspace=0.025)
    project_dirs = H._setup_dirs_()
    fig_name = 'figure_2_time_elon_profiles_{}.png'.format(ssw_event)
    fig_path = os.path.join(project_dirs['HUXt_figures'], fig_name)
    fig.savefig(fig_path)
    plt.close(fig)
    return


def load_metrics(ssw_event):
    """
    Function to load in the ensemble metrics (rmse, weight, arrival time etc), as well as compute the arrival and
    transit time of the deterministic huxt run and the SWPC cme data.
    Parameters
    ----------
    ssw_event: String identifier of which SWPC event to analyse. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    Returns
    -------
    metrics: pandas dataframe with the ensemble metrics in.
    t_arrive_det: Astropy Time object of the ConeCME arrival at Earth for the deterministic HUXt run.
    t_transit_det: Transit time of the ConeCME for the deterministic run.
    swpc_cme: Dictionary of the SWPC CME properties
    """
    
    swpc_cme = load_swpc_cme_forecasts(ssw_event)
    
    project_dirs = H._setup_dirs_()    
    metrics = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_event + "_ensemble_metrics.csv"))
    metrics['arrival_time'] = pd.to_datetime(metrics['arrival_time'])

    # Remove misses and reindex
    id_hit = metrics['hit'] == 1
    metrics = metrics.loc[id_hit, :]
    metrics.set_index(np.arange(0, metrics.shape[0]), inplace=True)

    # Renormalise weights
    metrics['w_a'] = metrics['w_a'] / metrics['w_a'].sum()
    metrics['w_b'] = metrics['w_b'] / metrics['w_b'].sum()
    metrics['w_avg'] = metrics['w_avg'] / metrics['w_avg'].sum()

    # Get the deterministic huxt run.
    path = os.path.join(project_dirs['HUXt_data'], "HUXt_*{}*_deterministic.hdf5".format(ssw_event))
    deterministic_run = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(deterministic_run)
    cme = cme_list[0]

    t_arrive_det = cme.earth_arrival_time
    t_transit_det = cme.earth_transit_time
    
    return metrics, t_arrive_det, t_transit_det, swpc_cme


def compute_ensemble_stats(ssw_event):
    """
    Compute statistics regarding the performance of the ensemble hindcasts, such as arrival time error and
    hindcast uncertaitny.
    Parameters
    ----------
    ssw_event: String identifier of the SWPC event being analysed. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    Returns
    -------
    ms: The pandas dataframe of ensemble metrics sorted by arrival time.
    stats: A dictionary of values including the hindcast uncertinaty, error, and skill for the weighted and unweighted
           unsemble hindcasts.
    """
    metrics, t_arrive_det, t_transit_det, swpc_cme = load_metrics(ssw_event)
    
    # Compute median, and IQR of ens avg, and weighted averages.
    # Sort by ensemble weight, then compute cumulative sum of weights.
    ms = metrics.sort_values(by='arrival_time', ascending=True)
    ms.set_index(np.arange(0, ms.shape[0]), inplace=True)
    ms['cwa'] = ms['w_a'].cumsum()
    ms['cwb'] = ms['w_b'].cumsum()
    ms['cwavg'] = ms['w_avg'].cumsum()
    
    arrival_error = np.abs(ms['arrival_time'] - pd.Timestamp(swpc_cme['t_arr_obs'].datetime))
    ms['error'] = arrival_error.dt.total_seconds() / 3600.0

    stats = {'obs': {'arrival': 0, 'error': np.NaN, 'skill': np.NaN},
             'swpc': {'arrival': 0, 'error': 0, 'skill': np.NaN},
             'det': {'arrival': 0, 'error': 0, 'skill': np.NaN},
             'ens': {'arrival': 0, 'lo': 0, 'hi': 0, 'error': 0, 'skill': 0},
             'cwa': {'arrival': 0, 'lo': 0, 'hi': 0, 'error': 0, 'skill': 0},
             'cwb': {'arrival': 0, 'lo': 0, 'hi': 0, 'error': 0, 'skill': 0},
             'cwavg': {'arrival': 0, 'lo': 0, 'hi': 0, 'error': 0, 'skill': 0}}

    # Quantile thresholds used to compute the ensemble spread - approxiamtes the 1-sigma error.
    q_lo = 0.16
    q_hi = 0.84

    stats['obs']['arrival'] = pd.Timestamp(swpc_cme['t_arr_obs'].datetime)
    stats['swpc']['arrival'] = pd.Timestamp(swpc_cme['t_arr_pred'].datetime)
    stats['swpc']['error'] = (stats['swpc']['arrival'] - stats['obs']['arrival']).total_seconds()/3600

    stats['det']['arrival'] = pd.Timestamp(t_arrive_det.datetime)
    stats['det']['error'] = (stats['det']['arrival'] - stats['obs']['arrival']).total_seconds()/3600
    stats['det']['skill'] = 1.0 - np.abs(stats['det']['error'] / stats['swpc']['error'])

    for key in ['ens', 'cwa', 'cwb', 'cwavg']:

        if key == 'ens':
            stats[key]['arrival'] = ms['arrival_time'].quantile(0.5)
            stats[key]['lo'] = ms['arrival_time'].quantile(q_lo)
            stats[key]['hi'] = ms['arrival_time'].quantile(q_hi)
        else:
            id_med = np.searchsorted(ms[key], 0.5)
            stats[key]['arrival'] = ms.loc[id_med, 'arrival_time']

            id_lo = np.searchsorted(ms[key], q_lo)
            stats[key]['lo'] = ms.loc[id_lo, 'arrival_time']

            id_hi = np.searchsorted(ms[key], q_hi)
            stats[key]['hi'] = ms.loc[id_hi, 'arrival_time']

        stats[key]['error'] = (stats[key]['arrival'] - stats['obs']['arrival']).total_seconds()/3600
        stats[key]['skill'] = 1.0 - np.abs(stats[key]['error'] / stats['swpc']['error'])
            
    return ms, stats


def print_stats_table(ssw_event):
    """
    Print to screen a latex formatted table of the ensemble hindscast performance statistics.
    Parameters
    ----------
    ssw_event: String identifier of the SWPC event being analysed. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    Returns
    -------
    """
    ms, stats = compute_ensemble_stats(ssw_event)
    fmt = "%Y-%m-%dT%H:%M"
    for key, val in stats.items():

        if key in ['ens', 'cwa', 'cwb', 'cwavg']:
            date = val['arrival'].strftime(fmt)
            lo = (val['arrival'] - val['lo']).total_seconds()/3600
            hi = (val['hi'] - val['arrival']).total_seconds()/3600
            rng = (val['hi'] - val['lo']).total_seconds()/3600
            err = val['error']
            print(key + " & {} & {:3.1f} (-{:3.1f}, +{:3.1f}) & {:3.1f}\\\\".format(date, rng, lo, hi, err))
        else:
            print(key + " & {} & & {:3.1f}\\\\".format(val['arrival'].strftime(fmt), val['error']))


def ensemble_histogram(ssw_event):
    """
    Produce a histogram of the ConeCME arrival times for the weighted and unweighted ensembles.
    Parameters
    ----------
    ssw_event: String identifier of the SWPC event being analysed. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    Returns
    -------
    A figure in the figures directory with name format: figure_3_arrival_distribution_{ssw_event}.png
    """
    metrics, t_arrive_det, t_transit_det, swpc_cme = load_metrics(ssw_event)
    ms, stats = compute_ensemble_stats(ssw_event)
    
    w = 20
    h = w / 3.0
    fig, ax = plt.subplots(1, 3, figsize=(w, h))

    t_lo = metrics['arrival_time'].min() - pd.Timedelta("2H")
    t_hi = metrics['arrival_time'].max() + pd.Timedelta("2H")
    bins = pd.date_range(t_lo.round('H'), t_hi.round('H'), freq="2H")

    t_arr_swpc = swpc_cme['t_arr_pred'].datetime
    t_arr_obs = swpc_cme['t_arr_obs'].datetime

    hist_colors = {'w_a': 'deepskyblue', 'w_b': 'violet', 'w_avg': 'sandybrown'}
    line_colors = {'w_a': 'mediumblue', 'w_b': 'darkviolet', 'w_avg': 'orangered'}
    labels = {'w_a': 'HI1A', 'w_b': 'HI1B', 'w_avg': 'HI1A+B'}
    for a, key, color in zip(ax, ['w_a', 'w_b', 'w_avg'], ['blue', 'green', 'orange']):
        a.hist(metrics['arrival_time'], bins, density=True, color='lightgrey', label='Ens. distribution')

        a.hist(metrics['arrival_time'], bins, weights=metrics[key], density=True, color=hist_colors[key], alpha=0.5,
               label=labels[key] + ' weighted  ens.')

        dxl = stats['ens']['arrival'] - stats['ens']['lo']
        dxu = stats['ens']['hi'] - stats['ens']['arrival']
        a.errorbar(stats['ens']['arrival'], 0.9, xerr=[[dxl], [dxu]], fmt='ks', label='Ens. median')

        if key == 'w_a':
            key2 = 'cwa'
        elif key == 'w_b':
            key2 = 'cwb'
        elif key == 'w_avg':
            key2 = 'cwavg'

        dxl = stats[key2]['arrival'] - stats[key2]['lo']
        dxu = stats[key2]['hi'] - stats[key2]['arrival']
        a.errorbar(stats[key2]['arrival'], 1.0, xerr=[[dxl], [dxu]], fmt='o', color=line_colors[key],
                   label='Weighted ens. median')

        a.vlines(t_arrive_det.datetime, 0, 3.5, colors='darkgreen', linewidths=2, linestyles=':', label='Deterministic')

        a.vlines(t_arr_swpc, 0, 3.5, colors='r', linewidths=2, linestyles='--', label='SWPC forecast')

        a.vlines(t_arr_obs, 0, 3.5, colors='k', linewidths=2, label='In-situ arrival')

        a.set_ylim(0, 3.5)
        a.set_xlabel('CME arrival time (day-hour)')

        hours = mpl.dates.HourLocator(interval=8)   # every hour
        hours_fmt = mpl.dates.DateFormatter('%d-%H')

        # format the ticks
        a.xaxis.set_major_locator(hours)
        a.xaxis.set_major_formatter(hours_fmt)

    for a in ax[1:]:
        a.set_yticklabels([])

    for a in ax:
        a.legend(loc='upper right', ncol=1, handletextpad=0.5, handlelength=1, columnspacing=0.5, fontsize=14,
                 framealpha=1.0)

    ax[0].set_ylabel('Ensemble density')

    fig.subplots_adjust(left=0.04, bottom=0.09, right=0.99, top=0.98, wspace=0.02)
    project_dirs = H._setup_dirs_()
    fig_name = 'figure_3_arrival_distribution_{}.png'.format(ssw_event)
    fig_path = os.path.join(project_dirs['HUXt_figures'], fig_name)
    fig.savefig(fig_path)
    plt.close(fig)
    return


def scatter_weight_error(ssw_event):
    """
    Produce a scatter plot of ensemble member weight versus arrival time error.
    Parameters
    ----------
    ssw_event: String identifier of the SWPC event being analysed. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    Returns
    -------
    A figure in the figures directory with name format: figure_4_error_vs_weight_{ssw_event}.png
    """
    ms, stats = compute_ensemble_stats(ssw_event)
    # How to add not about ensemles that don't hit?
    # Need to print stats about how many hit and miss in each event?
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(ms['w_a'], np.abs(ms['error']), 'o', color='mediumblue', label='H1A weighted')
    ax[1].plot(ms['w_b'], np.abs(ms['error']), 'o', color='darkviolet', label='H1B weighted')
    ax[2].plot(ms['w_avg'], np.abs(ms['error']), 'o', color='orangered', label='H1A+B weighted')

    ax[0].set_ylabel('Absolute arrival time error (hours)')

    whi = np.array([ms[key].max() for key in ['w_a', 'w_b', 'w_avg']])
    ehi = np.max(np.abs(ms['error'])) + 2
    for a in ax:
        a.set_xlabel('Ensemble member weight')
        a.set_xlim(0, whi.max())
        a.set_ylim(0, ehi)
        a.legend(loc='upper right', handletextpad=0.5, handlelength=1, fontsize=14, framealpha=1.0)

    for a in ax[1:]:
        a.set_yticklabels([])

    fig.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.99, wspace=0.05)
    project_dirs = H._setup_dirs_()
    fig_name = 'figure_4_error_vs_weight_{}.png'.format(ssw_event)
    fig_path = os.path.join(project_dirs['HUXt_figures'], fig_name)
    fig.savefig(fig_path)
    plt.close(fig)
    return


def animate_schematic(ssw_event):
    """
    Animate the model solution, and save as an MP4.
    Parameters
    ----------
    ssw_event: String identifier of the SWPC event being analysed. Should be ssw_007, ssw_008, ssw_009, ssw_012.

    Returns
    -------
    An MP4 in the figures directory with name HUXt_CR{cr_num}_{ssw_event}_movie.mp4
    """
    # Set the duration of the movie
    # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
    sim_duration = (1*u.day).to(u.s)
    movie_duration = sim_duration.value * (20 / 432000)
    
    # Load in the HUXT data and trace the flanks
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "HUXt_*{}*_deterministic.hdf5".format(ssw_event))
    deterministic_run = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(deterministic_run)
    cme = cme_list[0]
    hxt_sta, hxt_stb = huxt_t_e_profile_fast(cme)
    
    # LOAD IN THE CORRECT PA FROM THE SSW DATA FOR PLOTTING
    ssw_sta_name = ssw_event + "_sta.csv"
    ssw_sta_hxt = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_sta_name))
    ssw_stb_name = ssw_event + "_stb.csv"
    ssw_stb_hxt = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_stb_name))   
    
    sta_hxt_pa = ssw_sta_hxt.loc[0, 'pa']
    stb_hxt_pa = ssw_stb_hxt.loc[0, 'pa']
    print("{} STA PA of HUXt: {:3.2f}".format(ssw_event, sta_hxt_pa))
    print("{} STB PA of HUXt: {:3.2f}".format(ssw_event, stb_hxt_pa))
    del ssw_sta_hxt
    del ssw_stb_hxt
    
    times = model.time_init + model.time_out
    sta = H.Observer('STA', times)
    stb = H.Observer('STB', times)

    nt_end = np.sum(model.time_out <= sim_duration)
    
    def make_frame(t):
        """
        Produce the frame required by MoviePy.VideoClip.
        :param t: time through the movie
        """
        # Get the time index closest to this fraction of movie duration
        i = np.int32((nt_end - 1) * t / movie_duration)
        t_out = model.time_out[i]
        fig, ax = animation_frame(ssw_event, t_out, model, hxt_sta, hxt_stb, sta_hxt_pa, stb_hxt_pa, sta, stb)
        frame = mplfig_to_npimage(fig)
        plt.close('all')
        return frame

    cr_num = np.int32(model.cr_num.value)
    filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, ssw_event)
    filepath = os.path.join(project_dirs['HUXt_figures'], filename)
    animation = mpy.VideoClip(make_frame, duration=movie_duration)
    animation.write_videofile(filepath, fps=24, codec='libx264')
    return


def animation_frame(ssw_event, time, model, hxt_sta, hxt_stb, sta_hxt_pa, stb_hxt_pa, sta, stb):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    Parameters
    ----------
    ssw_event: String identifier of the SWPC event being analysed. Should be ssw_007, ssw_008, ssw_009, ssw_012.
    time: Time to look up closet model time to (with an astropy.unit of time).
    model: An instance of the HUXt class with a model solution.
    hxt_sta: Pandas dataframe of the ConeCME flank profile from HUXt for STEREO-A.
    hxt_stb: Pandas dataframe of the ConeCME flank profile from HUXt for STEREO-B.
    sta_hxt_pa: Float value of the position angle of the ConeCME in the STEREO-A field of view.
    stb_hxt_pa: Float value of the position angle of the ConeCME in the STEREO-B field of view.
    sta: A HUXt.Observer object for STEREO-A for this event.
    stb: A HUXt.Observer object for STEREO-B for this event.
    Returns
    -------
    fig: Figure handle.
    ax: List of the three Axes handles.
    """

    # Load in the deterministic HUXt solution, SSW data, and HI images.
    project_dirs = H._setup_dirs_()
        
    id_time_out = np.argmin(np.abs(model.time_out - time))
    time_out = model.time_init + model.time_out[id_time_out]

    # Load in the HI images
    hi1a_fc, hi1a_fp = find_hi_files(ssw_event, 'sta', time_out)
    hi1a_map = hip.get_image_diff(hi1a_fc, hi1a_fp, align=True, smoothing=True)

    hi1b_fc, hi1b_fp = find_hi_files(ssw_event, 'stb', time_out)
    hi1b_map = hip.get_image_diff(hi1b_fc, hi1b_fp, align=True, smoothing=True)

    # Load the SSW classifications for the HI images.
    img = 'diff'
    # Open up the SSW data
    ssw_out = tables.open_file(project_dirs['SSW_data'], mode="r")
    PLOT_HI1A = True
    PLOT_HI1B = True
    for craft, hi_map in zip(['sta', 'stb'], [hi1a_map, hi1b_map]):

        # Pull out event
        ssw_path = "/".join(['', ssw_event, craft, img])
        event = ssw_out.get_node(ssw_path)

        key = hi_map.date.strftime('T%Y%m%d_%H%M')+'01'
        
        # There is a bug in the naming/timing of the files. This is a workaround from sswanalysis   
        split = key.split('_')
        T_label = split[0]
        hhmm = split[1]
        if hhmm[2:] == '10':
            hhmm = '0901'
        if hhmm[2:] == '30':
            hhmm = '2901'
        if hhmm[2:] == '50':
            hhmm = '4901'
        
        key = T_label + '_' + hhmm
        
        try:
            cme_slice = event[key]

            # Get the CME front data
            cme = pd.DataFrame.from_records(cme_slice.cme_coords.read())
            cme.replace(to_replace=[99999], value=np.NaN, inplace=True)

            if craft == 'sta':
                ssw_sta = cme.copy()
            elif craft == 'stb':
                ssw_stb = cme.copy()
        except:
            if craft == 'sta':
                ssw_sta = None
                PLOT_HI1A = False
            elif craft == 'stb':
                ssw_stb = None
                PLOT_HI1B = False
                
    ssw_out.close()

    fig = plt.figure(figsize=(27, 10))
    
    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon = model.lon_grid.value.copy()
    rad = model.r_grid.value.copy()
    v = model.v_grid_cme.value[id_t, :, :].copy()
    
    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap = mpl.cm.viridis
    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    dv = 10
    levels = np.arange(200, 800+dv, dv)
    
    ax = plt.subplot(131, polar=True)
    ax2 = plt.subplot(132, polar=False)
    ax3 = plt.subplot(133, polar=False)
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries
    cme = model.cmes[0]
    ax.plot(cme.coords[id_t]['lon'], cme.coords[id_t]['r'], '-', color='darkorange', linewidth=3, zorder=3)

    for body, style in zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'], ['co', 'mo', 'ko', 'rs', 'y^']):
        obs = model.get_observer(body)
        if body != 'STB':
            ax.plot(obs.lon[id_t], obs.r[id_t], style, markersize=16, label=body)
        elif body == 'STB':
            ax.plot(obs.lon[id_t], obs.r[id_t], '^', color='fuchsia', markersize=16, label=body)
        
    #####################################################
    # Add on HI1A FOV, 4 to 25 
    rsa = np.mean(sta.r)
    lsa = np.mean(sta.lon)
    xsa = rsa * np.cos(lsa)
    ysa = rsa * np.sin(lsa)

    rsb = np.mean(stb.r)
    lsb = np.mean(stb.lon)
    xsb = rsb * np.cos(lsb)
    ysb = rsb * np.sin(lsb)
    
    sta_patch = [[lsa.value, rsa.value]]
    stb_patch = [[lsb.value, rsb.value]]

    # Get plane of sky coord of elon lims
    for el in [4.0, 24.0]:
        # STA
        rp = rsa * np.tan(el*u.deg)
        lp = lsa - 90*u.deg

        xp = rp * np.cos(lp)
        yp = rp * np.sin(lp)
        grad = (yp - ysa) / (xp - xsa)
        c = ysa - grad*xsa
        xf = 250 * u.solRad
        yf = grad*xf + c
        rf = np.sqrt(xf**2 + yf**2)
        lf = np.arctan2(yf, xf)
        sta_patch.append([lf.value, rf.value])

        # STB
        rp = rsb * np.tan(el*u.deg)
        lp = lsb + 90*u.deg

        xp = rp * np.cos(lp)
        yp = rp * np.sin(lp)
        grad = (yp - ysb) / (xp - xsb)
        c = ysb - grad*xsb
        xf = 250 * u.solRad
        yf = grad*xf + c
        rf = np.sqrt(xf**2 + yf**2)
        lf = np.arctan2(yf, xf)
        stb_patch.append([lf.value, rf.value])
        
    sta_patch = mpl.patches.Polygon(np.array(sta_patch), color='r', alpha=0.3, zorder=1)
    ax.add_patch(sta_patch)
    stb_patch = mpl.patches.Polygon(np.array(stb_patch), color='fuchsia', alpha=0.3, zorder=1)
    ax.add_patch(stb_patch)

    # Add on the flanks.
    id_t = np.argmin(np.abs(model.time_out - time))
    ax.plot([lsa.value, hxt_sta.loc[id_t, 'lon']], [rsa.value, hxt_sta.loc[id_t, 'r']], 'r-', linewidth=2)
    ax.plot(hxt_sta.loc[id_t,'lon'], hxt_sta.loc[id_t,'r'], 'rX', markersize=15, zorder=4)

    ax.plot([lsb.value, hxt_stb.loc[id_t, 'lon']], [rsb.value, hxt_stb.loc[id_t, 'r']], '-', color='fuchsia', linewidth=2)
    ax.plot(hxt_stb.loc[id_t, 'lon'], hxt_stb.loc[id_t, 'r'], 'd', color='fuchsia', markersize=15, zorder=4)
    ####################################################

    # Add on a legend.
    fig.legend(ncol=5, loc='lower left', bbox_to_anchor=(0.025, 0.0), frameon=False, handletextpad=0.2, columnspacing=1.0)

    ax.set_ylim(0, 240)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_facecolor('slategrey')
    fig.subplots_adjust(left=0.0, bottom=0.15, right=0.99, top=0.99, wspace=-0.12)
    
    # Add color bar
    pos = ax.get_position()
    dw = 0.005
    dh = 0.045
    left = pos.x0 + dw
    bottom = pos.y0 - dh
    wid = pos.width - 2*dw
    cbaxes = fig.add_axes([left, bottom, wid, 0.03])
    cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
    cbar1.set_label("Solar Wind Speed (km/s)")
    cbar1.set_ticks(np.arange(200, 900, 100))

    # Add label
    label = (model.time_init + model.time_out[id_t]).strftime("%Y-%m-%dT%H:%M")
    fig.text(0.25, pos.y0, label, fontsize=16)
    label = "HUXt2D"
    fig.text(0.07, pos.y0, label, fontsize=16)
    
    ###################################################
    normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
    
    for a, hi_map, ssw, PLOT_FLAG in zip([ax2, ax3], [hi1a_map, hi1b_map], [ssw_sta, ssw_stb], [PLOT_HI1A, PLOT_HI1B]):
        
        img = mpl.cm.gray(normalise(hi_map.data), bytes=True)

        # Plot out the raw frame
        if hi_map.observatory == "STEREO A":
            color = 'r'
            pa = sta_hxt_pa
            fmt = 'X'
        elif hi_map.observatory == "STEREO B":
            color = 'fuchsia'
            pa = stb_hxt_pa
            fmt = 'd'

        a.imshow(img, origin='lower')
        
        el_arr = np.arange(0, 25, 1)
        for dpa in [-2.0, 2.0]:
            pa_arr = np.zeros(el_arr.shape) + pa + dpa
            x, y = hip.convert_hpr_to_pix(el_arr*u.deg, pa_arr*u.deg, hi_map)
            a.plot(x, y, ':', color='lawngreen', linewidth=3)

        if PLOT_FLAG:
            a.plot(ssw['x'], ssw['y'], '-', color=color, linewidth=3 )
            a.plot(ssw['x_lo'], ssw['y_lo'], '--', color=color, linewidth=3)
            a.plot(ssw['x_hi'], ssw['y_hi'], '--', color=color, linewidth=3)

            id_pa = np.argmin(np.abs(ssw['pa'] - pa))
            a.plot(ssw.loc[id_pa, 'x'], ssw.loc[id_pa, 'y'], fmt, color=color, markersize=15)

        a.set_xlim(1, 1023)
        a.set_ylim(1, 1023)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    box_col = 'k'
    txt_x = 0.008
    txt_y = 0.96
    txt_fs = 18
    box = {'facecolor': box_col}
    label = "HI1A {}".format(hi1a_map.date.strftime("%Y-%m-%dT%H:%M"))
    ax2.text(txt_x, txt_y, label, transform=ax2.transAxes, fontsize=txt_fs, color='w', bbox=box)
    label = "HI1B {}".format(hi1b_map.date.strftime("%Y-%m-%dT%H:%M"))
    ax3.text(txt_x, txt_y, label, transform=ax3.transAxes, fontsize=txt_fs, color='w', bbox=box)
    return fig, [ax, ax2, ax3]


def print_table_1():
    """
    Print out a latex formatted table of each of hte SWPC forecast CME paramters and observations. Table 1 in the paper.
    """
    fmt = '%Y-%m-%dT%H:%M'
    cme_list = []
    for i, ssw_event in enumerate(['ssw_007', 'ssw_008', 'ssw_009', 'ssw_012']):
        cme = load_swpc_cme_forecasts(ssw_event)
        cme_list.append(cme)

    cme1 = cme_list[0]
    cme2 = cme_list[1]
    cme3 = cme_list[2]
    cme4 = cme_list[3]

    keys = ['event', 't_obs', 'lon', 'lat', 'width', 'v', 't_arr_pred', 't_arr_obs']
    label = {'t_obs': 'Time at 21.5~$R_{s}$', 'lon': 'Longitude (deg)', 'lat': 'Latitude (deg)', 'width':'Width (deg)',
             'v': 'Speed ($km~s^{-1}$)', 't_arr_pred': 'Predicted arrival', 't_arr_obs': 'Observed arrival', }
    for i, key in enumerate(keys):

        if key == 'event':
            line = "Event & CME 1 & CME 2 & CME 3 & CME 4 \\\\"
        elif key in ['t_obs', 't_arr_pred', 't_arr_obs']:
            fmt = "{} & {} & {} & {} & {} \\\\"
            line = fmt.format(label[key], cme1[key].strftime(fmt), cme2[key].strftime(fmt), cme3[key].strftime(fmt),
                              cme4[key].strftime(fmt))
        else:
            fmt = "{} & {:3.0f} & {:3.0f} & {:3.0f} & {:3.0f} \\\\"
            line = fmt.format(label[key], cme1[key].value, cme2[key].value, cme3[key].value, cme4[key].value)
        print(line)
        
        
def all_weight_vs_error_plots():
    """
    Function to produce a figure of all of the ensemble weight versus arrival error for all of the ssw events.
    Figure 6 in the paper.
    """
    fig, ax = plt.subplots(4, 3, figsize=(15, 20))

    w_lims = []
    e_lims = []
    for i, ssw_event in enumerate(['ssw_007', 'ssw_008', 'ssw_009', 'ssw_012']):

        ms, stats = compute_ensemble_stats(ssw_event)

        ax_sub = ax[i, :]
        ax_sub[0].plot(ms['w_a'], np.abs(ms['error']), 'o', color='mediumblue', label='H1A weighted')
        ax_sub[1].plot(ms['w_b'], np.abs(ms['error']), 'o', color='darkviolet', label='H1B weighted')
        ax_sub[2].plot(ms['w_avg'], np.abs(ms['error']), 'o', color='orangered', label='H1A+B weighted')

        w_hi = np.array([ms[key].max() for key in ['w_a', 'w_b', 'w_avg']])
        w_lims.append(w_hi.max())
        e_lims.append(np.max(np.abs(ms['error'])))

        n_hit = np.int32(np.sum(ms['hit']))
        txt_x = 0.8
        txt_y = 0.825
        txt_fs = 18
        cme_label = "CME-{:01d}".format(i+1)
        hit_label = "N={}".format(n_hit)
        for a in ax_sub:
            a.text(txt_x, txt_y, cme_label, transform=a.transAxes, fontsize=txt_fs)
            a.text(txt_x, txt_y-0.075, hit_label, transform=a.transAxes, fontsize=txt_fs)

    for a in ax.ravel():
        a.set_xlim(0, 0.045)
        a.set_ylim(0, 46)
        a.legend(loc='upper right', handletextpad=0.5, handlelength=1, columnspacing=0.5, fontsize=14, framealpha=1.0)

    for a in ax[-1, :]:
        a.set_xlabel('Ensemble member weight')

    for a in ax[:, 0]:
        a.set_ylabel('Absolute arrival time error (hours)')

    for a in ax[:, 1:].ravel():
        a.set_yticklabels([])

    for a in ax[0:3, :].ravel():
        a.set_xticklabels([])

    fig.subplots_adjust(left=0.05, bottom=0.03, right=0.99, top=0.99, wspace=0.025, hspace=0.025)
    project_dirs = H._setup_dirs_()
    fig_name = 'figure_3_all.png'
    fig_path = os.path.join(project_dirs['HUXt_figures'], fig_name)
    fig.savefig(fig_path)


def all_time_elongation_plots():
    """
    Function to produce a figure of all of the time elongation profiles for all of the ssw events.
    Figure 5 in the paper.
    """
    # Set up the figure
    w = 10
    h = 3*(w / 2.0)
    fig, ax_all = plt.subplots(4, 2, figsize=(w, h))

    for kk, ssw_event in enumerate(['ssw_007', 'ssw_008', 'ssw_009', 'ssw_012']):

        ax = ax_all[kk, :]

        # List the ensemble files, and set up space for results of comparisons
        project_dirs = H._setup_dirs_()
        path = os.path.join(project_dirs['HUXt_data'], "HUXt_*{}*_deterministic.hdf5".format(ssw_event))
        deterministic_run = glob.glob(path)[0]
        model, cme_list = H.load_HUXt_run(deterministic_run)
        cme = cme_list[0]

        hxta_det, hxtb_det = huxt_t_e_profile_fast(cme)

        # Load in the HUXt ensemble flanks
        sta_flanks = ssw_event + '_ensemble_sta.csv'
        ensemble_sta = pd.read_csv(os.path.join(project_dirs['out_data'], sta_flanks))
        stb_flanks = ssw_event + '_ensemble_stb.csv'
        ensemble_stb = pd.read_csv(os.path.join(project_dirs['out_data'], stb_flanks))
        # Load in the SSW profiles
        ssw_sta_name = ssw_event + "_sta.csv"
        ssw_sta = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_sta_name))
        ssw_stb_name = ssw_event + "_stb.csv"
        ssw_stb = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_stb_name)) 

        # Load in the metrics to find hits and misses
        project_dirs = H._setup_dirs_()    
        metrics = pd.read_csv(os.path.join(project_dirs['out_data'], ssw_event + "_ensemble_metrics.csv"))
        metrics['arrival_time'] = pd.to_datetime(metrics['arrival_time'])
        n_ens_all = metrics.shape[0]

        # Remove misses and reindex
        id_hit = np.argwhere(metrics['hit'] == 1)
        id_miss = np.argwhere(metrics['hit'] == 0)

        n_hit = len(id_hit)
        n_miss = len(id_miss)

        el_keys = []
        for col in ensemble_sta:
            if col[0:2] == 'el':
                el_keys.append(col)


        color = 'lightgrey'
        label = 'HUXt Ensemble (Hit)'
        ax[0].plot([], [], '-', color=color, zorder=1, label=label)
        ax[1].plot([], [], '-', color=color, zorder=1, label=label)

        color = 'moccasin'
        label = 'HUXt Ensemble (Miss)'
        ax[0].plot([], [], '-', color=color, zorder=1, label=label)
        ax[1].plot([], [], '-', color=color, zorder=1, label=label)

        for i, k in enumerate(el_keys):
            ta = ensemble_sta['time'] - model.time_init.jd
            tb = ensemble_stb['time'] - model.time_init.jd

            if i in id_hit:
                color = 'lightgrey'
            elif i in id_miss:
                color = 'moccasin'

            ax[0].plot(ta, ensemble_sta[k], '-', color=color, zorder=1)
            ax[1].plot(tb, ensemble_stb[k], '-', color=color, zorder=1)

        ta = hxta_det['time'] - model.time_init.jd
        ax[0].plot(ta, hxta_det['el'], '-', color='k', zorder=1, label='HUXt Deterministic')

        tb = hxtb_det['time'] - model.time_init.jd
        ax[1].plot(tb, hxtb_det['el'], '-', color='k', zorder=1, label='HUXt Deterministic')

        t = ssw_sta['time'] - model.time_init.jd
        ax[0].errorbar(t, ssw_sta['el'], yerr=[ssw_sta['el_dlo'], ssw_sta['el_dhi']], fmt='.', color='mediumblue',
                       zorder=2, label='HI1A SSW')

        t = ssw_stb['time'] - model.time_init.jd
        ax[1].errorbar(t, ssw_stb['el'], yerr=[ssw_stb['el_dlo'], ssw_stb['el_dhi']], fmt='.', color='darkviolet',
                       zorder=2, label='HI1B SSW')

        txt_x = 0.01
        txt_y = 0.93
        txt_fs = 18
        cme_label = "CME-{:01d}".format(kk+1)
        hit_label = "Hits: {}".format(n_hit)
        miss_label = "Misses: {}".format(n_miss)
        dy = 0.07
        for a in ax:
            a.text(txt_x, txt_y, cme_label, transform=a.transAxes, fontsize=txt_fs)
            a.text(txt_x, txt_y-dy, hit_label, transform=a.transAxes, fontsize=txt_fs)
            a.text(txt_x, txt_y-2*dy, miss_label, transform=a.transAxes, fontsize=txt_fs)

    for a in ax_all.ravel():
        a.set_xlim(-0.075, 1.9)
        a.set_ylim(4, 25)
        a.legend(loc='lower right', handletextpad=0.5, handlelength=1, columnspacing=0.5, fontsize=14, framealpha=1.0)

    for a in ax_all[-1, :]:
        a.set_xlabel('Time after CME onset (days)')

    for a in ax_all[:, 0]:
        a.set_ylabel('Elongation (degrees)')

    for a in ax_all[:, 1]:
        a.set_yticklabels([])

    for a in ax_all[0:3, :].ravel():
        a.set_xticklabels([])

    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.99, wspace=0.025, hspace=0.025)
    project_dirs = H._setup_dirs_()
    fig_name = 'figure_2_all.png'
    fig_path = os.path.join(project_dirs['HUXt_figures'], fig_name)
    fig.savefig(fig_path)


def print_summary_statistics():
    """
    Print out the percentage change in the hindcast arrival error and uncertainty for the combined HI1A and HI1B
    weighted ensemble relative to the unweighted ensemble
    """
    event_list = ['ssw_007', 'ssw_008', 'ssw_009', 'ssw_012']
    all_range = []
    all_error = []
    for event in event_list:

        ms, stats = compute_ensemble_stats(event)
        for key, val in stats.items():

            if key in ['ens', 'cwa', 'cwb', 'cwavg']:
                val['range'] = (val['hi'] - val['lo']).total_seconds()/3600

        range_change = 100*(stats['cwavg']['range'] - stats['ens']['range']) / stats['ens']['range']
        error_change = 100*(stats['cwavg']['error'] - stats['ens']['error']) / stats['ens']['error']
        all_range.append(range_change)
        all_error.append(error_change)
        print('*******************')
        print(event)
        print("Change in uncertainty: {:3.2f}".format(range_change))
        print("Change in error: {:3.2f}".format(error_change))

    all_range = np.array(all_range)
    avg_range = np.mean(all_range)
    sigma = np.std(all_range)
    serr_range = sigma / np.sqrt(all_range.size)

    all_error = np.array(all_error)
    avg_error = np.mean(all_error)
    sigma = np.std(all_error)
    serr_error = sigma / np.sqrt(all_error.size)

    print('*****')
    print("Mean change in uncertainty: {:3.1f} +/- {:3.1f}".format(avg_range, 2*serr_range))
    print("Mean change in error: {:3.1f} +/- {:3.1f}".format(avg_error, 2*serr_error))
    return
