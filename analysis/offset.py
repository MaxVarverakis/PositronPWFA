import h5py

def view_x(path, fname, species):
    with h5py.File(path + fname, 'r') as f:
        print(f[f'data/0/particles/{species}/position/x'][:])

def offset_x(path, fname, species, x_offset, show = False):
        if species not in ['recovery', 'witness']:
            print('Species must be either "recovery" or "witness"')
            return
        
        if show:
            view_x(path, fname, species)
        with h5py.File(path + fname, 'r+') as f:
            f[f'data/0/particles/{species}/position/x'][:] += x_offset
        if show:
            view_x(path, fname, species)



if __name__ == '__main__':
    spec = 'bW'
    
    path = f'/Users/max/HiPACE/recovery/filament/h5/offsets/{spec}/'
    fname = 'openpmd_000000.h5'

    # NOTE Treat this like a loaded weapon... only uncomment when you're ready to fire
    
    # offset_x(path, fname, 'recovery', .025, True)
    # offset_x(path, fname, species = 'witness', x_offset = 0.0145, show = True)