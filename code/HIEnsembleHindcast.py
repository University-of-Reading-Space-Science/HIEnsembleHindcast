import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
import sunpy.coordinates.sun as sn
import tables
import pandas as pd
from stereo_spice.coordinates import StereoSpice
import scipy.interpolate as interp
import os 
import HUXt as H
import glob

spice = StereoSpice()

def huxt_t_e_profile(model):
    """
    Compute the time elonagation profile of the flank of a ConeCME in HUXt, from either the STEREO-A or STEREO-B persepctive.
    """
    times =  model.time_init + model.time_out
    ert = H.Observer('EARTH', times)
    sta = H.Observer('STA', times)
    stb = H.Observer('STB', times)

    sta_profile = pd.DataFrame(index=np.arange(model.time_out.size), columns=['time', 'el', 'r', 'lon', 'pa'])
    stb_profile = pd.DataFrame(index=np.arange(model.time_out.size), columns=['time', 'el', 'r', 'lon', 'pa'])

    sta_profile['time'] = times.jd
    stb_profile['time'] = times.jd
    cme = model.cmes[0]

    for i, coord in cme.coords.items():

        if len(coord['r'])==0:
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

            
def compute_arrival_time(model, cme, arr_lon, arr_rad):
    """
    Function to compute arrival time at a set longitude and radius of the CME front.
    
    Tracks radial distance of front along a given longitude out past specified longitude.
    Then interpolates the r-t profile to find t_arr at arr_rad. 
    """
    
    # Need to force units to be the same to make interpolations work 
    arr_lon = arr_lon.to(model.lon.unit)
    arr_rad = arr_rad.to(model.r.unit)
    
    # Check if hit or miss.
    # Put longitude between -180 - 180, centered on CME lon.
    lon_diff = arr_lon - cme.longitude
    if lon_diff < -180*u.deg:
        lon_diff += 360*u.deg
    elif lon_diff > 180*u.deg:
        lon_diff -= 360*u.deg
        
    cme_hw = cme.width/2.0
    if (lon_diff >= -cme_hw) & (lon_diff <= cme_hw):
        # HIT, so get t-r profile along lon of interest.
        t_front = []
        r_front = []

        for i, coord in cme.coords.items():

            if len(coord['r'])==0:
                continue

            t_front.append(model.time_out[i].to('d').value)

            # Lookup radial coord at earth lon
            r = coord['r'].value
            lon = coord['lon'].value

            # Only keep front of cme
            id_front = r > np.mean(r)
            r = r[id_front]
            lon = lon[id_front]
            
            r_ans = np.interp(arr_lon.value, lon, r, period=model.twopi)
            r_front.append(r_ans)
            # Stop when max r 
            if r_ans > arr_rad.value:
                break

        t_front = np.array(t_front)
        r_front = np.array(r_front)

        t_transit = np.interp(arr_rad.value, r_front, t_front)
        t_transit = t_transit*u.d
        t_arrival = model.time_init + t_transit
    else:
        t_transit = np.NaN*u.d
        t_arr = np.NaN*u.d

    return t_arrival, t_transit

def compute_rmse(hxt, ssw):
    """
    Function to compute root mean square error between the huxt and ssw time elongation profiles of the CME flank.
    """
    elon_interp = np.interp(ssw['time'].values, hxt['time'].values, hxt['el'].values, left=np.NaN, right=np.NaN)
    hxt_interp = pd.DataFrame({'time':ssw['time'].values, 'el':elon_interp})
    de = (hxt_interp['el'] - ssw['el'])**2
    n_rms_samp = np.sum(np.isfinite(de))
    rms = np.sqrt(de.mean(skipna=True))
    return rms, n_rms_samp

def get_ssw_profile(ssw_event, craft, img, pa_center, pa_wid=1.0):
    """
    Load the Solar Stormwatch profile of the CME for either the HI1A or HI1B data. 
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

def plot_huxt_and_hi_schematic(model, time, hi1a_map, hi1b_map, ssw_sta, ssw_stb, sta, stb):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.

    :param time: Time to look up closet model time to (with an astropy.unit of time). 
    :return fig: Figure handle.
    :return ax: Axes handle.
    """
    fig = plt.figure(figsize=(27,10))
    
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
    #fig, ax = plt.subplots(figsize=(30, 10), subplot_kw={"projection": "polar"})
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
        
    sta_patch = mpl.patches.Polygon(np.array(sta_patch),color='r', alpha=0.3, zorder=1)
    ax.add_patch(sta_patch)
    stb_patch = mpl.patches.Polygon(np.array(stb_patch),color='fuchsia', alpha=0.3, zorder=1)
    ax.add_patch(stb_patch)

    # Add on the flanks.
    id_t = np.argmin(np.abs(model.time_out - t_plot))
    ax.plot([lsa.value, hxt_sta.loc[id_t,'lon']], [rsa.value, hxt_sta.loc[id_t,'r']], 'r-', linewidth=2)
    ax.plot(hxt_sta.loc[id_t,'lon'], hxt_sta.loc[id_t,'r'], 'rX', markersize=15, zorder=4)

    ax.plot([lsb.value, hxt_stb.loc[id_t,'lon']], [rsb.value, hxt_stb.loc[id_t,'r']], '-', color='fuchsia', linewidth=2)
    ax.plot(hxt_stb.loc[id_t,'lon'], hxt_stb.loc[id_t,'r'], 'd', color='fuchsia', markersize=15, zorder=4)
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
    fig.text(0.25, pos.y0, label , fontsize=16)
    label = "HUXt2D"
    fig.text(0.07, pos.y0, label, fontsize=16)
    
    ###################################################
    normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)

    for a, hi_map, ssw in zip([ax2, ax3], [hi1a_map, hi1b_map], [ssw_sta, ssw_stb]):
        img = mpl.cm.gray(normalise(hi_map.data), bytes=True)

        # Plot out the raw frame
        if hi_map.observatory == "STEREO A":
            color = 'r'
            pa = 89
            fmt = 'X'
        elif hi_map.observatory == "STEREO B":
            color = 'fuchsia'
            pa = 270
            fmt = 'd'

        a.imshow(img, origin='lower')

        a.plot(ssw['x'], ssw['y'], '-', color=color, linewidth=3 )
        a.plot(ssw['x_lo'], ssw['y_lo'], '--', color=color, linewidth=3)
        a.plot(ssw['x_hi'], ssw['y_hi'], '--', color=color, linewidth=3)

        el_arr = np.arange(0,25,1)
        for dpa in [-2.0, 2.0]:
            pa_arr = np.zeros(el_arr.shape) + pa + dpa
            x, y = hip.convert_hpr_to_pix(el_arr*u.deg, pa_arr*u.deg, hi_map)
            a.plot(x, y, ':', color='lawngreen', linewidth=3)

        id_pa = np.argmin(np.abs(ssw['pa'] - pa))
        a.plot(ssw.loc[id_pa, 'x'], ssw.loc[id_pa, 'y'], fmt, color=color, markersize=15)

        a.set_xlim(1, 1023)
        a.set_ylim(1, 1023)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    box_col='k'
    txt_x=0.008
    txt_y=0.96
    txt_fs=18
    box = {'facecolor': box_col}
    label= "HI1A {}".format(hi1a_map.date.strftime("%Y-%m-%dT%H:%M"))
    ax2.text(txt_x, txt_y, label, transform=ax2.transAxes, fontsize=txt_fs, color='w', bbox=box)
    label= "HI1B {}".format(hi1b_map.date.strftime("%Y-%m-%dT%H:%M"))
    ax3.text(txt_x, txt_y, label, transform=ax3.transAxes, fontsize=txt_fs, color='w', bbox=box)
    
    project_dirs = H._setup_dirs_()
    fig_name = os.path.join(project_dirs['figures'], 'figure_1_huxt_and_hi.png')
    fig.savefig(fig_name)
    return fig, [ax, ax2, ax3]
