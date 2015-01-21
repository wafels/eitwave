from sim import wave2d
import astropy.units as u
import numpy as np
import os
from sunpy.map import Map

m2deg = 360. / (2 * 3.1415926 * 6.96e8)

params = {
    "cadence": 12., #seconds

    "hglt_obs": 0. * u.degree, #degrees
    "rotation": 360. / (27. * 86400.) * u.degree / u.s, #degrees/s, rigid solar rotation

    #Wave parameters that are initial conditions
    "direction": -205. * u.degree, #degrees, measured CCW from HG +latitude
    "epi_lat": 15. * u.degree, #degrees, HG latitude of wave epicenter
    "epi_lon": 9. * u.degree, #degrees, HG longitude of wave epicenter

    #Wave parameters that can evolve over time
    #The first element is constant in time
    #The second element (if present) is linear in time
    #The third element (if present) is quadratic in time
    #Be very careful of non-physical behavior
    "width": np.asarray([90., 1.5]) * u.degree, #degrees, full angle in azimuth, centered at 'direction'
    "wave_thickness": np.asarray([6.0e6 * m2deg, 6.0e4*m2deg]) * u.degree, #degrees, sigma of Gaussian profile in longitudinal direction
    "wave_normalization": [1.], #integrated value of the 1D Gaussian profile
    "speed": np.asarray([9.33e5 * m2deg, -1.495e3 * m2deg]) * u.degree / u.s, #degrees/s, make sure that wave propagates all the way to lat_min for polynomial speed

    #Random noise parameters
    "noise_type": "Poisson", #can be None, "Normal", or "Poisson"
    "noise_scale": 0.1,
    "noise_mean": 1.,
    "noise_sdev": 1.,

    #Structured noise parameters
    "struct_type": "None", #can be None, "Arcs", or "Random"
    "struct_scale": 5.,
    "struct_num": 10,
    "struct_seed": 13092,

    "max_steps": 20,

    "clean_nans": True,

    #HG grid, probably would only want to change the bin sizes
    "lat_min": -90. * u.degree,
    "lat_max": 90. * u.degree,
    "lat_bin": 0.2 * u.degree,
    "lon_min": -180. * u.degree,
    "lon_max": 180. * u.degree,
    "lon_bin": 5. * u.degree,

    #HPC grid, probably would only want to change the bin sizes
    "hpcx_min": -1228.8 * u.arcsec,
    "hpcx_max": 1228.8 * u.arcsec,
    "hpcx_bin": 2.4 * u.arcsec,
    "hpcy_min": -1228.8 * u.arcsec,
    "hpcy_max": 1228.8 * u.arcsec,
    "hpcy_bin": 2.4 * u.arcsec
}

def test_wave2d(write=None, max_steps=20):

    params[max_steps] = max_steps

    #wave_maps = wave2d.simulate(params)
    wave_maps = wave2d.simulate(params, verbose = True)
    
    #To get simulated HG' maps (centered at wave epicenter):
    #wave_maps_raw = wave2d.simulate_raw(params)
    #wave_maps_raw_noise = wave2d.add_noise(params, wave_maps_raw)
    
    #visualize(wave_maps)
    
    """
    import util
    
    new_wave_maps = []
    
    for wave in wave_maps:
        print("Unraveling map at "+str(wave.date))
        new_wave_maps += [util.map_hpc_to_hg_rotate(wave, epi_lon = 45., epi_lat = 30., xbin = 5, ybin = 0.2)]
    
    
    from matplotlib import colors
    
    wave_maps_raw = wave2d.simulate_raw(params)
    wave_maps_transformed = wave2d.transform(params, wave_maps_raw, verbose = True)
    
    #First simulation slide
    wave_maps_raw[19].show()
    wave_maps_transformed[19].show()
    
    #Second simulation slide
    wave_maps[19].show(norm = colors.Normalize(0,1))
    new_wave_maps[19].show(norm = colors.Normalize(0,1))
    """
    if write is not None:
        for im, m in enumerate(wave_maps):
            fname = os.path.join(os.path.expanduser(write), 'simulated_euv_wave_%05d.fits' % im)
            m.save(fname)

    return Map(wave_maps, cube=True)
