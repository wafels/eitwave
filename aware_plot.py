#
# Output plots for AWARE.
#

import numpy as np
import matplotlib.pyplot as plt
import os


# Elementary string formatting
fmt = '%.1f'

# Format a result string
def _result_string(x, ex, s1, s2):
    __string = fmt + '\pm' + fmt
    _string = __string % (x, ex)
    return s1 + _string + s2


#
# Plot out summary dynamics for all the arcs
#
def all_arcs_summary_plots(dynamics, imgdir, example, simulated_params=None):
    """
    :param dynamics:
    :param imgdir:
    :param example:
    :param simulated_params:
    :return:
    """

    position_choice = dynamics[0][0].positiom_choice
    error_choice = dynamics[0][0].error_choice

    # Plot all the arcs
    plt.figure(1)
    for r in dynamics:
        if r[1].fitted:
            plt.plot(r[1].timef, r[1].locf)
    plt.xlabel('time since originating event')
    plt.ylabel('degrees of arc from originating event [%s, %s]' % (position_choice, error_choice))
    plt.title(example + ': wavefront locations')
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.savefig(os.path.join(imgdir, example + '_%s_%s_arcs.png' % (position_choice, error_choice)))

    # Plot all the projected arcs
    plt.figure(2)
    for r in dynamics:
        if r[1].fitted:
            p = np.poly1d(r[1].quadfit)
            time = np.arange(0, r[0].times[-1])
            plt.plot(time, p(time))
    plt.xlabel('time since originating event')
    plt.ylabel('degrees of arc from originating event [%s, %s]' % (position_choice, error_choice))
    plt.title(example + ': best fit arcs')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(os.path.join(imgdir, example + '_%s_%s_best_fit_arcs.png' % (position_choice, error_choice)))


    # Plot the estimated velocity at the original time t = 0
    plt.figure(3)
    v = []
    ve = []
    arcindex = []
    notfitted =[]
    for ir, r in enumerate(dynamics):
        if r[1].fitted:
            v.append(r[1].velocity)
            ve.append(r[1].velocity_error)
            arcindex.append(ir)
        else:
            notfitted.append(ir)
    plt.errorbar(arcindex, v, yerr=(ve, ve), fmt='ro', label='estimated original velocity')
    if len(notfitted) > 0:
        plt.axvline(notfitted[0], linestyle=':', label='not fitted')
        for nf in notfitted[1:]:
            plt.axvline(nf, linestyle=':')
    if simulated_params is not None:
        plt.axhline(simulated_params["true_velocity"], label='true velocity')
    plt.xlabel('arc number')
    plt.ylabel('estimated original velocity (km/s) [%s, %s]' % (position_choice, error_choice))
    plt.legend(framealpha=0.5)
    plt.title(example + ': estimated original velocity across wavefront')
    plt.savefig(os.path.join(imgdir, example + '_%s_%s_initial_velocity.png' % (position_choice, error_choice)))


    # Plot the estimated acceleration
    plt.figure(4)
    a = []
    ae = []
    arcindex = []
    notfitted =[]
    for ir, r in enumerate(dynamics):
        if r[1].fitted:
            a.append(r[1].acceleration)
            ae.append(r[1].acceleration_error)
            arcindex.append(ir)
        else:
            notfitted.append(ir)
    plt.errorbar(arcindex, a, yerr=(ae, ae), fmt='ro', label='estimated acceleration')
    if len(notfitted) > 0:
        plt.axvline(notfitted[0], linestyle=':', label='not fitted')
        for nf in notfitted[1:]:
            plt.axvline(nf, linestyle=':')
    if simulated_params is not None:
        plt.axhline(simulated_params["true_acceleration"], label='true acceleration')
    plt.xlabel('arc number')
    plt.ylabel('estimated acceleration (m/s/s) [%s, %s]' % (position_choice, error_choice))
    plt.title(example + ': estimated acceleration across wavefront')
    plt.legend(framealpha=0.5)
    plt.savefig(os.path.join(imgdir, example + '_%s_%s_acceleration.png' % (position_choice, error_choice)))

    return None


#
# Plot the fit of the wavefront position as a function of time, along a
# single arc.
#
def fitposition(fp):

    # Plot all the data
    plt.scatter(fp.times,
                fp.pos,
                label='measured wave location (%s, %s)' % (fp.position_choice, fp.error_choice),
                marker='.', c='b')

    # Plot the data that was assessed to be fitable
    plt.errorbar(fp.timef,
                 fp.locf,
                 yerr=(fp.errorf, fp.errorf),
                 fmt='ro',
                 label='fitted data')

    plt.xlim(0.0, fp.times[-1])
    # Locations of fit results printed as text on the plot
    tpos = 0.5 * (fp.times[1] - fp.times[0]) + fp.times[0]
    ylim = plt.ylim()
    ylim = [0.5 * (ylim[0] + ylim[1]), ylim[1]]
    lpos = ylim[0] + np.arange(1, 4) * (ylim[1] - ylim[0]) / 4.0

    # Plot the results of the fit process.
    if fp.fitted:
        plt.plot(fp.timef,
                 fp.bestfit, label='best fit')
        # Strings describing the fit parameters
        acc_string = _result_string(1000 * fp.acceleration, 1000 * fp.acceleration_error, "$a=", '\, m s^{-2}$')
        v_string = _result_string(fp.velocity, fp.velocity_error, "$v=", '\, km s^{-1}$')
        plt.text(tpos, lpos[0], acc_string)
        plt.text(tpos, lpos[1], v_string)
        plt.axvline(fp.offset, linestyle=":", label='first measurement (time=%.1f)' % fp.offset, color='k')
    else:
        plt.text(tpos, ylim[1], 'fit failed')

    # Label the plot
    plt.xlabel('time since originating event (seconds)')
    plt.ylabel('degrees of arc from first measurement')
    plt.legend(framealpha=0.5, loc=4)
    plt.show()


#
# Plot a summary the wavefront along a given arc, as a function of
# time.
#
def arc(tarc):
    plt.imshow(tarc.data, aspect='auto',
               extent=[tarc.times[0], tarc.times[-1],
                       tarc.latitude[0], tarc.latitude[-1]])
    plt.xlim(0, tarc.times[-1])
    _label = 'first data point (time=' + fmt + ')'
    plt.axvline(tarc.offset, label=_label % tarc.offset, color='w')
    plt.fill_betweenx([tarc.latitude[0], tarc.latitude[-1]],
                      tarc.offset, hatch='X', facecolor='w', label='not observed')
    plt.ylabel('degrees of arc from first measurement')
    plt.xlabel('time since originating event (seconds)')
    plt.title('arc' + tarc.title)
    plt.legend()
    plt.show()


#
# This summarizes and plots the results for many trials of the simulated wave.
#
def swave_summary_plots(imgdir, filename, results, params):

    # Number of trials
    ntrial = len(results)

    # Number of arcs
    narc = len(results[0])

    # Storage for the results
    fitted = np.zeros((ntrial, narc))
    v = np.zeros_like(fitted)
    ve = np.zeros_like(fitted)
    a = np.zeros_like(fitted)
    ae = np.zeros_like(fitted)
    qmean = np.zeros(narc)
    nfound = np.zeros(narc)

    # Indices of all the arcs
    all_arcindex = np.arange(0, narc)

    # Quantity plots
    qcolor = 'r'
    fmt = qcolor + 'o'

    # Number of trials with a successful fit.
    nfcolor = 'b'

    #
    # Recover the information we need
    #
    for itrial, dynamics in enumerate(results):

        # Go through each arc and get the results
        for ir, r in enumerate(dynamics):
            if r[1].fitted:
                fitted[itrial, ir] = True
                v[itrial, ir] = r[1].velocity
                ve[itrial, ir] = r[1].velocity_error
                a[itrial, ir] = r[1].acceleration
                ae[itrial, ir] = r[1].acceleration_error
            else:
                fitted[itrial, ir] = False

    #
    # Make the velocity and acceleration plots
    #
    for j in range(0, 2):
        fig, ax1 = plt.subplots()

        # Select which quantity to plot
        if j == 0:
            q = v
            qe = ve
            qname = 'velocity'
            initial_value = params['speed'][0].value
        else:
            q = a
            qe = ae
            qname = 'acceleration'
            initial_value = params['speed'][1].value

        # Initial values to get the plot legend labels done
        arcindex = np.nonzero(fitted[0, :])
        qerr = qe[0, arcindex]
        ax1.errorbar(arcindex, q[0, arcindex], yerr=(qerr, qerr),
                     fmt=fmt, label='estimated %s' % qname)
        # Plot the rest of the values found.
        for i in range(1, ntrial):
            arcindex = np.nonzero(fitted[1, :])
            qerr = qe[i, arcindex]
            ax1.errorbar(arcindex, q[i, arcindex], yerr=(qerr, qerr), fmt=fmt)

        # Mean quantity over all the trials
        for i in range(0, narc):
            f = fitted[:, i]
            trialindex = np.nonzero(f)
            nfound[i] = np.sum(f)
            qmean[i] = np.mean(q[trialindex, :])

        # Plot the mean values found
        ax1.plot(all_arcindex, qmean, label='mean %s' % qname)

        # Plot the line that indicates the true velocity at t=0
        ax1.axhline(initial_value, label='true %s' % qname)

        # Labelling the quantity plot
        ax1.set_xlabel('arc index')
        ax1.set_ylabel('estimated %s (km/s)' % qname)
        ax1.legend(framealpha=0.5)
        ax1.title('%s: estimated %s across wavefront' % (qname, params['name']))
        for tl in ax1.get_yticklabels():
            tl.set_color(qcolor)

        # Plot the fraction
        ax2 = ax1.twinx()
        ax2.plot(all_arcindex, nfound / np.float64(ntrial),
                 label='fraction of trials fitted', color=nfcolor)
        ax2.set_ylabel('fraction of trials fitted [%i trials]' % ntrial,
                       color=nfcolor)
        for tl in ax2.get_yticklabels():
            tl.set_color(nfcolor)

        # Save the figure
        plt.savefig(os.path.join(imgdir, '%s_initial_%s.png' % (filename, qname)))
