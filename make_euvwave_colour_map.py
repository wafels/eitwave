
from copy import deepcopy
import numpy as np
import sunpy.map
import matplotlib.pyplot as plt
import aware, aware_utils

def make_euvwave_colour_map(transformed,mc,example,info,result,thresh=15.0):

#two inputs
#transformed - this is the cleaned running difference persistence map (RDPI) cube
#mc - this is the original map cube

    startind = {'previous1':9,
                'corpita_fig4':1,
                'corpita_fig6':0,
                'corpita_fig8a':3,
                'corpita_fig8e':3,
                'corpita_fig7':3}

    n=startind['corpita_fig8e']
#mask = np.zeroes_like(transformed[0])
    
    mask = deepcopy(transformed[0])
    mask.data = mask.data * 0.0
    counter = 1
    tims = []
    for h in transformed[n:]:
        ind = np.where(h.data > thresh)
        mask.data[ind] = counter
        tims.append(h.meta['t_obs'][11:19])
        counter = counter + 1

    ind2 = np.where(mask.data == 0.0)
    mask.data[ind2] = np.nan

    cbar_tickvals = np.int32(np.arange(1,counter-1,3))
    #cbar_ticklabels = (np.array(cbar_tickvals) + n).tolist()
   # timescale = accum * 12

#plt.figure(7)


    #now do some tricks to plot an arc on the top.
    params = aware_utils.params(result[info[example]['result']])
    arcmask = aware.unravel([mc[5]],params)
    arcmask[0].data[:] = 0.0
    arcmask[0].data[:,2] = 1.0
    arcmap = aware_utils.map_reravel([arcmask[0]],params)
    arcmap2 = arcmap[0]
    inds = np.where(arcmap2.data == 0.0)
    arcmap2.data[inds] = np.nan

    #now make the composite map for the plot!
    compmap = sunpy.map.Map(mc[5],mask,composite=True)#arcmap2,composite=True)
    #plt.clim([0,20000])
    compmap.set_colors(1,'nipy_spectral')
    compmap.set_colors(0,'gray_r')
    compmap.set_alpha(1,0.8)
    #compmap.set_colors(2,'binary')
    #compmap.set_alpha(2,0.6)
    
    
    figure = plt.figure()
    axes = figure.add_subplot(111)
    ret = compmap.plot(axes=axes)
    compmap.draw_limb()
    compmap.draw_grid()
    #plt.text(result[info[example]['result']]['hpc_x'],result[info[example]['result']]['hpc_y'],'x',fontsize=12,color='r')
    plt.plot(result[info[example]['result']]['hpc_x'],result[info[example]['result']]['hpc_y'],'ro')
    plt.clim([0,10000])
    
    cbar = figure.colorbar(ret[1], ticks=cbar_tickvals )#[2,4,6,8,10,12,14,16,18,20,22,24,26])

    #print cbar_tickvals
    #print cbar_ticklabels
    #print len(tims)
    tims2=[]
    #need to get labels right
    for u in cbar_tickvals:
        tims2.append(tims[u])

    cbar.ax.set_yticklabels(tims2) #[cbar_ticklabels[:])#   (tims[0],tims[2],tims[4],tims[6],tims[8],tims[10]
                     #   ,tims[12],tims[14],tims[16],tims[18],tims[20], tims[22],tims[24]))
    plt.title('AWARE detection ' + info[example]['tr'].start.split(' ')[0] )
    plt.savefig('euvwave_contour_map_'+example + '.eps')
    figure.show()
