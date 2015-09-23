#
# Create some simulated wave data from the input parameters.
#
from sim import wave2d

def simulate_wave2d(params=None, max_steps=20, verbose=True,
                    output=['finalmaps']):

    # To get simulated HG' maps (centered at wave epicenter):
    # wave_maps_raw = wave2d.simulate_raw(params)
    # wave_maps_raw_noise = wave2d.add_noise(params, wave_maps_raw)

    # wave_maps = wave2d.simulate(params)
    return wave2d.simulate(params, max_steps, verbose=verbose,
                           output=output)
