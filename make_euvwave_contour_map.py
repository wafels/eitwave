

def make_euvwave_colour_map(transformed,mc,thresh=20.0):

#two inputs
#transformed - this is the cleaned running difference persistence map (RDPI) cube
#mc - this is the original map cube

#mask = np.zeroes_like(transformed[0])
    
    mask = deepcopy(transformed[0])
    mask.data = np.zeroes_like(mask.data) #mask.data * 0.0
    counter = 1
    tims = []
    for h in transformed[3:]:
        ind = np.where(h.data > thresh)
        mask.data[ind] = counter
        tims.append(h.meta['t_obs'][11:19])
        counter = counter + 1

    ind2 = np.where(mask.data == 0.0)
    mask.data[ind2] = np.nan
    
    timescale = accum * 12

#plt.figure(7)

    compmap = sunpy.map.Map(mc[5],mask,composite=True)
    #plt.clim([0,20000])
    compmap.set_colors(1,'nipy_spectral')
    compmap.set_alpha(1,0.8)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    ret = compmap.plot(axes=axes)
    plt.clim([0,20000])
    cbar = figure.colorbar(ret[1], ticks=[2,4,6,8,10,12,14,16,18,20,22])

    #need to get labels right

    cbar.ax.set_yticklabels((tims[0],tims[2],tims[4],tims[6],tims[8],tims[10]
                        ,tims[12],tims[14],tims[16],tims[18],tims[20]))
    plt.title('AWARE detection')
    plt.savefig('euvwave_contour_map.pdf')
    figure.show()
