import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_obs_tracking(particle, weight, true_state, obs_image, plot_batch = 30, shown_particle=True, title = 'UDPF'):

    prediction = torch.sum(particle*weight[:,:,:,None], dim = 2).detach().cpu().numpy()

    imaget = obs_image.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    (batch_size, seq_len, num_particle, state_dim) = particle.shape

    state = true_state.detach().cpu().numpy()

    particle = particle.detach().cpu().numpy()
    weight = weight.detach().cpu().numpy()

    # plot config
    legend_size = 40
    axis_label_size = 15
    title_size = 40
    tick_params_size = 20

    head_scale = 1.5
    quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 20., 'width': 0.003, 'headlength': 5 * head_scale,
                   'headwidth': 1 * head_scale, 'headaxislength': 4.5 * head_scale}
    marker_kwargs = {'markersize': 9, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}

    # reverse the color, then it tracks the Cyan
    batch = plot_batch

    plt.figure()

    for t in range(seq_len):
        plt.gca().clear()
        plt.title(f't={t}'+ title, fontsize=axis_label_size)
        plt.imshow(np.ones((128,128,3)) - imaget[batch, t], origin='lower')

        x = state[batch, t, 0] + 64  # change the origin
        y = state[batch, t, 1] + 64
        vx = state[batch, t, 2]
        vy = state[batch, t, 3]
        plt.quiver(x, y, vx, vy, color='red')
        plt.plot(x, y, 'ro')

        if shown_particle:
            ppx = particle[batch, t, :, 0] + 64
            ppy = particle[batch, t, :, 1] + 64
            vvx = particle[batch, t, :, 2]
            vvy = particle[batch, t, :, 3]
            plt.quiver(ppx, ppy, vvx, vvy, color='black')
            # plt.quiver(ppx, ppy, vvx, vvy, weight[batch, t, :])
            alpha_value = (weight[batch, t, :]-(weight[batch, t, :].min()))/(weight[batch, t, :].max()-(weight[batch, t, :].min()))
            plt.scatter(ppx, ppy, s =weight[batch, t, :]*10000, c= 'blue', alpha= alpha_value)

        px = prediction[batch, t, 0] + 64  # change the origin
        py = prediction[batch, t, 1] + 64
        vpx = prediction[batch, t, 2]
        vpy = prediction[batch, t, 3]
        plt.plot(px, py, 'b*')
        plt.quiver(px, py, vpx, vpy, color='blue')

        plt.xticks(np.arange(0, 128.5, 20), fontsize=axis_label_size)
        plt.yticks(np.arange(0, 128.5, 20), fontsize=axis_label_size)

        plt.xlim([0, 128.1])
        plt.ylim([0, 128.1])

        #     plt.savefig('./Fig3/timestep'+str(t)+'.pdf')
        plt.show()

        plt.pause(0.5)


def plot_state_tracking(particle, weight, true_state, loss, plot_batch = 30, title = 'UDPF'):
    prediction = torch.sum(particle * weight[:, :, :, None], dim=2).detach().cpu().numpy()

    state = true_state.detach().cpu().numpy()

    # plot
    legend_size = 15
    axis_label_size = 15
    title_size = 15
    tick_params_size = 20

    head_scale = 1.5
    quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 20., 'width': 0.003, 'headlength': 5 * head_scale,
                   'headwidth': 1 * head_scale, 'headaxislength': 4.5 * head_scale}
    marker_kwargs = {'markersize': 9, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}

    plt.figure()

    for batch in range(plot_batch, plot_batch+1):
        plt.gca().clear()
        # print(batch)
        x_3 = state[batch, :, 0] + 64  # change the origin
        y_3 = state[batch, :, 1] + 64
        # vx=state[batch,:,2]
        # vy=state[batch,:,3]
        # plt.quiver(x,y,vx,vy,color='red')
        f1, = plt.plot(x_3, y_3, '-r')

        # x_20=state_2[batch,0,0]+64 # change the origin
        # y_20=state_2[batch,0,1]+64
        # # vx=state[batch,:,2]
        # # vy=state[batch,:,3]
        # # plt.quiver(x,y,vx,vy,color='red')
        # plt.plot(x_20,y_20,'ro')

        # ppx=particle[batch,t,:,0]+64
        # ppy=particle[batch,t,:,2]+64
        # vvx=particle[batch,t,:,2]
        # vvx=particle[batch,t,:,3]
        #     plt.plot(ppx,ppy,'bo')

        px_3 = prediction[batch, :, 0] + 64  # change the origin
        py_3 = prediction[batch, :, 1] + 64
        # vpx=prediction[batch,t,2]
        # vpy=prediction[batch,t,3]
        f2, = plt.plot(px_3, py_3, '-b')
        # plt.quiver(px,py,vpx,vpy,color='blue')

        plt.xticks(np.arange(0, 128.5, 20), fontsize=axis_label_size)
        plt.yticks(np.arange(0, 128.5, 20), fontsize=axis_label_size)

        plt.xlim([0, 128.1])
        plt.ylim([0, 128.1])

        plt.title(title + ' with val loss %.3f' %loss, fontsize=title_size)
        plt.legend([f1, f2], ['true state', 'predicted state'], fontsize=legend_size)

        # plt.savefig('./UDPF(feature).pdf')
        plt.show()


def plot_ess_tracking(weight, loss, plot_batch = 30, title='UDPF'):
    # plot
    legend_size = 15
    axis_label_size = 15
    title_size = 15
    tick_params_size = 20

    (batch_size, seq_len, num_particle) = weight.shape

    weight = weight.detach().cpu().numpy()

    plt.figure()
    for batch in range(plot_batch, plot_batch+1):
        plt.gca().clear()
        ESS = 1 / np.sum(weight[batch, :, :] ** 2, axis=-1)

        f2, = plt.plot(np.arange(seq_len), ESS, '-b')

        plt.title(title + ':ESS' +' with val loss %.3f' %loss, fontsize=title_size)

        # plt.savefig('./UDPF(feature).pdf')
        plt.show()


def plot_motion_model(outputs, iteration, plot_batch = 30, title = 'Pretrain_dyn'):
    (s_pred, state) = outputs

    prediction = s_pred.detach().cpu().numpy()

    state = state.detach().cpu().numpy()

    # plot
    legend_size = 15
    axis_label_size = 15
    title_size = 15
    tick_params_size = 20

    head_scale = 1.5
    quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 20., 'width': 0.003, 'headlength': 5 * head_scale,
                   'headwidth': 1 * head_scale, 'headaxislength': 4.5 * head_scale}
    marker_kwargs = {'markersize': 9, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}

    plt.figure()

    for batch in range(plot_batch, plot_batch + 1):
        plt.gca().clear()
        # print(batch)
        x_3 = state[batch, :, 0] + 64  # change the origin
        y_3 = state[batch, :, 1] + 64
        # vx=state[batch,:,2]
        # vy=state[batch,:,3]
        # plt.quiver(x,y,vx,vy,color='red')
        f1, = plt.plot(x_3, y_3, '-r')

        # x_20=state_2[batch,0,0]+64 # change the origin
        # y_20=state_2[batch,0,1]+64
        # # vx=state[batch,:,2]
        # # vy=state[batch,:,3]
        # # plt.quiver(x,y,vx,vy,color='red')
        # plt.plot(x_20,y_20,'ro')

        # ppx=particle[batch,t,:,0]+64
        # ppy=particle[batch,t,:,2]+64
        # vvx=particle[batch,t,:,2]
        # vvx=particle[batch,t,:,3]
        #     plt.plot(ppx,ppy,'bo')

        px_3 = prediction[batch, :, 0] + 64  # change the origin
        py_3 = prediction[batch, :, 1] + 64
        # vpx=prediction[batch,t,2]
        # vpy=prediction[batch,t,3]
        f2, = plt.plot(px_3, py_3, '-b')
        # plt.quiver(px,py,vpx,vpy,color='blue')

        plt.xticks(np.arange(0, 128.5, 20), fontsize=axis_label_size)
        plt.yticks(np.arange(0, 128.5, 20), fontsize=axis_label_size)

        plt.xlim([0, 128.1])
        plt.ylim([0, 128.1])

        plt.title(title + str(iteration), fontsize=title_size)
        plt.legend([f1, f2], ['true state', 'predicted state'], fontsize=legend_size)

        # plt.savefig('./UDPF(feature).pdf')
        plt.show()
        plt.cla()

    plt.close()

def plot_obs(img, reconstruct_img, plot_batch = 30):

    imaget = img.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    reconstruct_imgt = reconstruct_img.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    plt.figure()

    # plot time step: 0, 24, 49
    timestep = np.array([0, 19, 29, 39])
    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.imshow(imaget[plot_batch, timestep[i]])

        plt.subplot(2, 4, i+1+4)
        plt.imshow(reconstruct_imgt[plot_batch, timestep[i]])

    plt.show()

    plt.close()