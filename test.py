import pytec_fn
import util_fn
import os
import json
from skimage import io, img_as_float
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

start = time.time()

roi = pytec_fn.define_roi()
iw = 128

plot_fig_cmap = 1
plot_vec = 1
plot_stats = 1
save_files = 1

# read the data json
with open('project_metadata.json') as f:
    project_data = json.load(f)
    project = project_data['project']
    project = project[0]
    project_path = project['project_path']
    project_name = project['project_name']

    calibration = project_data['calibration']
    calibration = calibration[0]
    im_width = calibration['im_width']
    im_height = calibration['im_height']

    source = project_data['source']
    source = source[1]
    n_exp_img = source['number_experiment_images']

# main loop
for f in range(1, n_exp_img):

    gaussian_sigma = 64
    kernel_normal = 64

    f1 = img_as_float(io.imread(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'IMG_'+str(f)+'.tif')))
    f2 = img_as_float(io.imread(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'IMG_' + str(f+1) + '.tif')))

    f1p = pytec_fn.pre_proc(f1, gaussian_sigma, kernel_normal)
    f2p = pytec_fn.pre_proc(f2, gaussian_sigma, kernel_normal)

    grid_x_flat, grid_y_flat, n_iw_x, n_iw_y, n_iw = pytec_fn.split_2frames(iw, roi)
    C, P, PP, PP2, M, S, SNR = pytec_fn.cor_2frames(f1p, f2p, iw, grid_x_flat, grid_y_flat, n_iw)

    grid_x = np.reshape(grid_x_flat, (int(n_iw_y), int(n_iw_x)))
    grid_y = np.reshape(grid_y_flat, (int(n_iw_y), int(n_iw_x)))

    if plot_fig_cmap == 1:

        export_folder = os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'EXPORT')

        t = os.path.exists(export_folder)
        if t:
            counter = 1
            while t:
                new_export_folder = export_folder + '_' + str(counter)
                t = os.path.exists(new_export_folder)
                counter += 1

        else:
            new_export_folder = export_folder

        os.mkdir(new_export_folder)

        k = int(np.floor(n_iw_x * np.floor(n_iw_y/2)) + np.floor(n_iw_x/2))

        fig = plt.figure(figsize=(12, 9))
        ax_0 = fig.add_subplot(1, 1, 1)
        ax_0.imshow(f1, cmap='gray')
        ax_0.scatter(grid_x, grid_y, marker='+', s=5, linewidth=0.5, color='yellow', zorder=1000)
        ax_0.scatter(grid_x_flat[k], grid_y_flat[k], marker='o', s=20, linewidth=1.0, edgecolor='green',
                     facecolor='none', zorder=2000)
        ax_0.add_patch(
            plt.Rectangle((grid_x_flat[k] - iw / 2, grid_y_flat[k] - iw / 2), iw, iw, linewidth=0.75, edgecolor='b',
                          facecolor='none'))
        ax_0.add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=0.5, edgecolor='r', facecolor='none'))
        ax_0.tick_params(labelsize=7)

        fig.canvas.draw()
        fig.savefig('ROI_PTS' + str(f) + '_' + str(k) + '.png', bbox_inches='tight')
        plt.pause(1)
        plt.close(fig)

        fig1 = plt.figure(figsize=(5, 3))
        ax1_0 = fig1.add_subplot(1, 1, 1)
        im1 = ax1_0.imshow(C[:, :, k], cmap='viridis')
        ax1_0.tick_params(labelsize=7)
        util_fn.colorbar(im1)
        fig1.canvas.draw()
        fig1.savefig(os.path.join(new_export_folder, 'cmap_' + str(f) + '_' + str(k) + '.pdf'), bbox_inches='tight')
        plt.pause(1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(6, 4))
        ax2_0 = fig2.add_subplot(1, 1, 1)
        SNR_g = np.reshape(SNR, (int(n_iw_y), int(n_iw_x)))
        im2 = ax2_0.imshow(SNR_g)
        ax2_0.tick_params(labelsize=7)
        util_fn.colorbar(im2)
        fig2.canvas.draw()
        fig2.savefig(os.path.join(new_export_folder,'SNR-' + str(f) + '.pdf'), bbox_inches='tight')
        plt.pause(1)
        plt.close(fig2)

    dx_i_f = PP2[1, :] - (iw / 2)  # position of the peak in integer value
    dy_i_f = PP2[0, :] - (iw / 2)

    dx_i_g = np.reshape(dx_i_f, (int(n_iw_y), int(n_iw_x)))  # reshaped to the size of the grid
    dy_i_g = np.reshape(dy_i_f, (int(n_iw_y), int(n_iw_x)))

    sub_dx_f, sub_dy_f = pytec_fn.cor_sub_pix(C, PP2, n_iw)

    sub_dx_g = np.reshape(sub_dx_f, (int(n_iw_y), int(n_iw_x)))  # reshaped relative position of the peak
    sub_dy_g = np.reshape(sub_dy_f, (int(n_iw_y), int(n_iw_x)))

    dx = dx_i_g + sub_dx_g
    dy = dy_i_g + sub_dy_g

    if plot_vec == 1:
        N = 1
        xr = grid_x[::N, ::N]
        yr = grid_y[::N, ::N]
        dxr = dx[::N, ::N]
        dyr = dy[::N, ::N]

        mm = np.max(np.sqrt(dxr.flatten() * dxr.flatten() + dyr.flatten() * dyr.flatten()))
        ddx = iw / 2  # distance between grid points in pix

        v_scale = -0.9 * (ddx*N) / mm

        fig3 = plt.figure(figsize=(7, 7))
        ax3_0 = fig3.add_subplot(1, 1, 1)
        ax3_0.imshow(f1, cmap='gray')
        ax3_0.quiver(xr, yr, v_scale*dxr, v_scale*dyr, angles='xy', scale_units='xy',
                       scale=1, headwidth=3, headlength=3, headaxislength=3,
                       width=0.002, pivot='middle', color='orange')
        ax3_0.set_aspect('equal')
        ax3_0.tick_params(labelsize=7)
        fig3.canvas.draw()
        fig3.savefig(os.path.join(new_export_folder,'vectors_map-' + str(f) + '.pdf'), bbox_inches='tight')
        plt.pause(1)
        plt.close(fig3)

        fig4 = plt.figure(figsize=(4, 4))
        ax4 = fig4.add_subplot(1, 1, 1)
        ax4.plot(-1 * dx.flatten(), -1 * dy.flatten(), '+', color='slategrey')
        ax4.set_aspect('equal')
        ax4.set_ylabel('dy', fontsize=7)
        ax4.set_xlabel('dx', fontsize=7)
        ax4.tick_params(labelsize=7)
        fig4.canvas.draw()
        fig4.savefig(os.path.join(new_export_folder,'cloud_map-' + str(f) + '.pdf'), bbox_inches='tight')
        plt.pause(1)
        plt.close(fig4)

        fig5 = plt.figure(figsize=(8, 4))
        ax5 = fig5.add_subplot(1, 2, 1)

        ax5.hist(-1 * dx.flatten(), bins=20)
        ax5.set_ylabel('count', fontsize=7)
        ax5.set_xlabel('dx', fontsize=7)
        ax5.tick_params(labelsize=7)

        ax6 = fig5.add_subplot(1, 2, 2)
        ax6.hist(-1 * dy.flatten(), bins=20)
        ax6.set_ylabel('count', fontsize=7)
        ax6.set_xlabel('dx', fontsize=7)
        ax6.tick_params(labelsize=7)
        fig5.canvas.draw()
        fig5.savefig(os.path.join(new_export_folder,'histogram-' + str(f) + '.pdf'), bbox_inches='tight')
        plt.pause(1)
        plt.close(fig5)

    if save_files == 1:
        t = os.path.exists(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'VECTORS'))

        if not t:
            os.mkdir(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'VECTORS'))

        filename = os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'VECTORS', 'vector_' + str(f) + '.h5')
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('x', data=grid_x)
            hf.create_dataset('y', data=grid_y)
            hf.create_dataset('Dx', data=dx)
            hf.create_dataset('Dy', data=dy)

    print('image pair ' + str(f) + '/10 done: ' + str(100 * f / 10) + ' %')

end = time.time()
print(' ')
print('> Processing done in '+str(end - start)+' seconds')

