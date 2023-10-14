import defs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import constants

plt.rc('text', usetex = True)
plt.rcParams['figure.figsize'] = [10.0, 6.0]
plt.rcParams['font.size'] = 18

def x_centroidPlot(class_name, show, save_name = ''):
    plt.plot(class_name.witnessInsitu["time"] * class_name.kp_inv, class_name.driveInsitu['average']['[x]'], label = 'Drive Beam', c = 'r')
    plt.plot(class_name.witnessInsitu["time"] * class_name.kp_inv, class_name.witnessInsitu['average']['[x]'], label = 'Witness Beam', c = 'b')
    plt.plot(class_name.witnessInsitu["time"] * class_name.kp_inv, class_name.recoveryInsitu['average']['[x]'], label = 'Recovery Beam', c = 'g')
    plt.xlabel('Distance Propagated [m]')
    plt.ylabel('$k_p\\bar{x}$')
    plt.legend()
    plt.xlim(0, .2)
    
    if save_name:
        plt.savefig(save_name)

    if show:
        plt.show()
    else:
        plt.close()
    

def frames(steps, prop_dist, frame_dir, path, insitu_path, n0, normalized, src_path, regime):
    # ffmpeg -framerate 30 -i frames_%04d.png fbWOff.mp4
    
    if regime == 'filament':
        prof_scale = 1.5
        ylim = 6

        for j in range(steps + 1):
            frameData = defs.Functions(path = path, insitu_path = insitu_path, n0 = n0, iteration = j, normalized = normalized, recovery = True, src_path=src_path)
            
            ax = plt.axes()
            im = plt.pcolormesh(frameData.info.z, frameData.info.x, frameData.ExmBy.T, cmap = 'RdBu') #, vmin = -1, vmax = 1)
            plt.title(f'Propagated Distance: {j * prop_dist / steps:.2f} cm')

            plt.pcolormesh(frameData.info.z, frameData.info.x, frameData.jz_beam.T * frameData.IA, cmap = 'RdBuT', vmin = -1e15, vmax = 1e15)
            plt.plot(frameData.info.z, prof_scale * frameData.profile / max(frameData.profile) - ylim, 'k', alpha = .5)

            plt.xlim(frameData.info.zmin, frameData.info.zmax)
            plt.ylim(-ylim, ylim)
            plt.ylabel('$k_px$')
            plt.xlabel('$k_p\zeta$')

            ax2 = plt.twinx()
            ax2.plot(frameData.info.z, frameData.Ez, color = 'm')
            ax2.set_ylim(-.8, .8)
            ax2.set_ylabel(r'$E_z/E_0$',  labelpad = 1, color = 'm')
            ax2.spines["right"].set_color('m')
            ax2.tick_params(axis='y', colors='m')

            divider2 = make_axes_locatable(ax)
            cax2 = divider2.append_axes("right", size = "4%", pad = .9)
            divider3 = make_axes_locatable(ax2)
            cax3 = divider3.append_axes("right", size = "4%", pad = .9)
            cax3.remove()


            cb2 = plt.colorbar(im, cax = cax2)
            # cb2.formatter.set_useMathText(True)
            # cb2.formatter.set_powerlimits((0, 0))
            cb2.set_label(r'$(E_x - B_y)/E_0 $')

            # plt.show()

            plt.savefig(f'{frame_dir}/frame_{j:04}.png', dpi = 500, bbox_inches = 'tight')
            plt.close()
    
    elif regime == 'uniform':
        prof_scale = .75
        ylim = 3
        s = .6

        for j in range(steps + 1):
            frameData = defs.Functions(path = path, insitu_path = insitu_path, n0 = n0, iteration = j, normalized = normalized, recovery = True, src_path=src_path)
            
            ax = plt.axes()
            im = plt.pcolormesh(frameData.kp * frameData.info.z, frameData.kp * frameData.info.x, frameData.ExmBy.T / frameData.E0, cmap = 'RdBu', vmin = -s, vmax = s)
            plt.title(f'Propagated Distance: {j * prop_dist / steps:.2f} cm')
            
            plt.pcolormesh(frameData.kp * frameData.info.z, frameData.kp * frameData.info.x, frameData.jz_beam.T, cmap = 'RdBuT', vmin = -1e12, vmax = 1e12)
            plt.plot(frameData.kp * frameData.info.z, prof_scale * frameData.profile / max(frameData.profile) - ylim, 'k', alpha = .5)

            ax2 = plt.twinx()
            ax2.plot(frameData.kp * frameData.info.z, frameData.Ez / frameData.E0, color = 'm')
            ax2.set_ylim(-.5, .5)
            ax2.set_ylabel(r'$E_z/E_0$',  labelpad = 1, color = 'm')
            ax2.spines["right"].set_color('m')
            # ax2.spines["left"].set_visible(False)
            ax2.tick_params(axis='y', colors='m')
            divider2 = make_axes_locatable(ax)
            cax2 = divider2.append_axes("right", size = "4%", pad = .9)
            divider3 = make_axes_locatable(ax2)
            cax3 = divider3.append_axes("right", size = "4%", pad = .9)
            cax3.remove()
            cb2 = plt.colorbar(im, cax = cax2)
            cb2.set_label(r'$(E_x - B_y)/E_0 $')
            
            ax.set_xlim(frameData.kp * frameData.info.zmin, frameData.kp * frameData.info.zmax)
            ax.set_xlabel(r'$k_p\xi$')


            ax.set_ylim(-3, 3)
            ax.set_ylabel(r'$k_px$')

            # plt.show()

            plt.savefig(f'{frame_dir}/frame_{j:04}.png', dpi = 500, bbox_inches = 'tight')
            plt.close()


if __name__ == '__main__':
    
    n = True

    regime = 'filament'
    # regime = 'uniform'

    if regime == 'filament':
        ne_plasma = 1e17 # cm^-3
        d = 'behindOffset/recovery'
    elif regime == 'uniform':
        ne_plasma = 7.8e15 # cm^-3
        d = 'behind'
    
    p = f'/Users/max/HiPACE/recovery/{regime}/h5/{d}/'
    ip = f'/Users/max/HiPACE/recovery/{regime}/insitu/{d}/'

    data = defs.Functions(path = p, insitu_path = ip, n0 = ne_plasma, iteration = 0, normalized = n, recovery = True, src_path='/Users/max/HiPACE')
    data.customCMAP()
    
    # x_centroidPlot(data, True, save_name='')

    # frames(steps=50, prop_dist=.2, frame_dir='/Users/max/Downloads', path=p, insitu_path=ip, n0=ne_plasma, normalized=n, src_path='/Users/max/HiPACE', regime=regime)



