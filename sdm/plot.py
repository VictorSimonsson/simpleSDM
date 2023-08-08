import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy import signal


def plot_rotation(coords, title, length=0.1):
    # Create a scatter plot
    plt.scatter(coords[:, 0], coords[:, 1])
    lim = 3
    # Set the x and y-axis labels
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.title(title)

    plt.show()


def polar_plot(ir_data, doa_data):
    # Convert directions of arrival to radians
    phi = np.deg2rad(doa_data[:, 1])  # Azimuth
    theta = np.deg2rad(90 - doa_data[:, 0])  # Elevation (assuming the z-axis is the vertical axis)

    # Calculate magnitude of the impulse response
    mag = np.abs(ir_data)

    # Create polar plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    # Plot magnitude vs. time and direction of arrival
    for i in range(ir_data.shape[0]):  # Loop over the rows (time samples)
        ax.plot(theta, mag[i, :], label=f'Time {i+1}')

    # Set labels and title
    ax.set_title('Impulse Response Magnitude vs. Time and Direction of Arrival')
    ax.set_xlabel('Elevation (degrees)')
    ax.set_ylabel('Impulse Response Magnitude')
    ax.legend()

    plt.show()


def doa(azi, colat, fs, win_len, p=None, size=250, title=''):
    """Direction of Arrival, with optional p(t) scaling the siz
    """
    # t in ms
    t_ms = np.linspace(0, len(azi) / fs, len(azi), endpoint=False) * 1000

    # shift azi to [np.pi, np.pi]
    azi[azi > np.pi] = azi[azi > np.pi] % -np.pi
    # colat to elevation
    ele = np.pi/2 - colat

    if p is not None:
        s_plot = np.clip(p / np.max(p), 10e-15, None)
    else:
        s_plot = np.ones_like(azi)
    # scale
    s_plot *= size

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')

    # plot in reverse order so that first reflections are on top
    p = ax.scatter(azi[:], ele[:], s=s_plot[::-1], c=t_ms[:],
                   alpha=0.35)
    ax.invert_xaxis()
    ax.set_xlabel("Azimuth in rad")
    ax.set_ylabel("Elevation in rad")
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi / 2$', r'$0$',
                        r'$\pi / 2$', r'$\pi$'])

    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\pi / 2$', r'$0$', r'$\pi / 2$'])
    ax.set_title(title)
    # show t as colorbar
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
    cbar.set_label("t in ms")

    try:
        # produce a legend with a cross section of sizes from the scatter
        handles, labels = p.legend_elements(prop="sizes", alpha=0.3, num=5,
                                            func=lambda x: x/size)
        ax.legend(handles, labels, loc="upper right", title="p(t)")
    except AttributeError:  # mpl < 3.3.0
        pass
    plt.show()
    filename = f"DOA_plot_Fs{48000}_WinLen{win_len}.png"
    filepath = "Data/Plots/"
    plt.savefig(filepath+filename)


def doa3(azi, colat, fs, win_len, p=None, size=250, title='', meas=''):
    """Direction of Arrival, with optional p(t) scaling the siz
    """
    # t in ms
    t_ms = np.linspace(0, len(azi) / fs, len(azi), endpoint=False) * 1000

    # shift azi to [np.pi, np.pi]
    azi[azi > np.pi] = azi[azi > np.pi] % -np.pi
    # colat to elevation
    ele = np.pi/2 - colat

    if p is not None:
        s_plot = np.clip(p / np.max(p), 10e-15, None)
    else:
        s_plot = np.ones_like(azi)
    # scale
    s_plot *= size

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')

    # plot in reverse order so that first reflections are on top
    p = ax.scatter(azi[:], ele[:], s=s_plot[::-1], c=t_ms[:],
                   alpha=0.35)
    ax.invert_xaxis()
    ax.set_xlabel("Azimuth in degrees")
    ax.set_ylabel("Elevation in degrees")
    # Convert ticks to degrees
    xticks_deg = np.arange(-180, 181, 30)
    yticks_deg = np.arange(-90, 91, 30)

    # Convert ticks to radians
    xticks_rad = np.deg2rad(xticks_deg)
    yticks_rad = np.deg2rad(yticks_deg)

    # Set the ticks and tick labels
    ax.set_xticks(xticks_rad)
    ax.set_xticklabels([f"{angle}" for angle in xticks_deg])
    ax.set_yticks(yticks_rad)
    ax.set_yticklabels([f"{angle}" for angle in yticks_deg])

    ax.set_title(title)
    # show t as colorbar
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
    cbar.set_label("t in ms")

    try:
        # produce a legend with a crossection of sizes from the scatter
        handles, labels = p.legend_elements(prop="sizes", alpha=0.3, num=5,
                                            func=lambda x: x/size)
        ax.legend(handles, labels, loc="upper right", title="p(t)")
    except AttributeError:  # mpl < 3.3.0
        pass
    plt.show()
    filename = f"DOA_plot_Fs_{title}_meas.png"
    filepath = "Data/Plots/"
    plt.savefig(filepath + filename, transparent=False)


def plot_doa(coordinates, pressure_signal, sampling_frequency):
    # Convert Cartesian coordinates to spherical coordinates
    r = np.linalg.norm(coordinates, axis=1)
    theta = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    phi = np.arccos(coordinates[:, 2] / r)

    # Calculate time based on sampling frequency
    time = np.linspace(0, pressure_signal.shape[1] / sampling_frequency, pressure_signal.shape[1])

    # Create the scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(theta, phi, r, c=time, s=pressure_signal, cmap='viridis')

    # Set plot labels and colorbar
    ax.set_xlabel('Azimuth (theta)')
    ax.set_ylabel('Elevation (phi)')
    ax.set_zlabel('Distance (r)')
    cbar = fig.colorbar(sc, ax=ax, label='Time')

    # Show the plot
    plt.show()


def doa2(azi, colat, fs, win_len, p=None, size=250, title=''):
    """Direction of Arrival, with optional p(t) scaling the size."""
    # t in ms
    t_ms = np.linspace(0, len(azi) / fs, len(azi), endpoint=False) * 1000

    # shift azi to [np.pi, np.pi]
    azi[azi > np.pi] = azi[azi > np.pi] % -np.pi
    # colat to elevation
    ele = np.pi/2 - colat

    if p is not None:
        s_plot = np.clip(p / np.max(p), 10e-15, None)
        max_index = np.argmax(p)  # Find the index of the maximum value in p
        max_azi = azi[max_index]  # Get the corresponding azimuth
        max_ele = ele[max_index]  # Get the corresponding elevation
    else:
        s_plot = np.ones_like(azi)
        max_azi = None
        max_ele = None

    # scale
    s_plot *= size
    #a = utils.normalize(s_plot)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal')

    # plot in reverse order so that first reflections are on top
    p = ax.scatter(azi[::-1], ele[::-1], s=s_plot[::-1], c=t_ms, alpha=0.4)

    if max_azi is not None and max_ele is not None:
        ax.plot(max_azi, max_ele, 'rs', markersize=10)  # Plot the maximum DOA as a red square

    ax.invert_xaxis()
    ax.set_xlabel("Azimuth in rad")
    ax.set_ylabel("Elevation in rad")
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi / 2$', r'$0$', r'$\pi / 2$', r'$\pi$'])
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r'$-\pi / 2$', r'$0$', r'$\pi / 2$'])
    ax.set_title(title)

    # Add grid lines at every 10 degrees
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(np.deg2rad(range(-180, 181, 45)))
    ax.set_yticks(np.deg2rad(range(-90, 91, 45)))
    ax.set_xticklabels(range(-180, 181, 45))
    ax.set_yticklabels(range(-90, 91, 45))

    # show t as colorbar
    cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
    cbar.set_label("t in ms")
    plt.show()


def plot_polar_dB(pressure, DOA_cartesian, min_dB=-40, max_dB=0):
    # Convert Cartesian coordinates to polar coordinates
    magnitude, angle = np.abs(DOA_cartesian), np.angle(DOA_cartesian, deg=True)

    # Convert pressure to dB scale
    dB = 20 * np.log10(pressure)

    # Normalize dB values
    normalized_dB = (dB - min_dB) / (max_dB - min_dB)

    # Convert angle to radians
    radians = np.deg2rad(angle)

    # Plot polar plot
    plt.figure()
    plt.polar(radians, normalized_dB, marker='o')

    # Customize plot appearance
    plt.thetagrids(range(0, 360, 45))  # Set angle gridlines
    plt.rgrids(np.arange(0, 1.1, 0.2), angle=0, weight=0.5)  # Set radial gridlines
    plt.ylim(0, 1)  # Set radial axis limits
    plt.title('Polar Plot in dB')
    plt.show()



def plot_polar_dB2(pressure, DOA_cartesian, min_dB=-40, max_dB=0):
    # Convert Cartesian coordinates to polar coordinates
    magnitude, angle = np.abs(DOA_cartesian), np.angle(DOA_cartesian, deg=True)

    # Convert pressure to dB scale
    dB = 20 * np.log10(pressure)

    # Normalize dB values
    normalized_dB = (dB - min_dB) / (max_dB - min_dB)

    # Convert angle to radians
    radians = np.deg2rad(angle)

    # Plot polar plot
    plt.figure()
    plt.polar(radians, normalized_dB, marker='o')

    # Customize plot appearance
    plt.thetagrids(range(0, 360, 45))  # Set angle gridlines
    plt.rgrids(np.arange(0, 1.1, 0.2), angle=90, weight=0.5)  # Set radial gridlines (angle=90 for top-down view)
    plt.ylim(0, 1)  # Set radial axis limits
    plt.title('Polar Plot in dB')
    plt.show()


def plot_coords(DOA, ir, locs):
    # Suppress/hide the warning
    np.seterr(invalid='ignore')
    r = np.linalg.norm(DOA, axis=1)  # radial distance
    ele = np.nan_to_num(np.arccos(DOA[:, 2] / r)) # Elevation angle (theta)
    azi = np.arctan2(DOA[:, 1], DOA[:, 0])  # azimuthal angle (phi)
    mag = ir / np.max(ir)  # magnitude of each coordinate
    # Plot all samples except the first one
    plt.scatter(np.degrees(azi[1:]), np.degrees(ele[1:]), s=mag[1:] * 100, c=mag[1:], cmap='coolwarm', alpha=0.5)

    # Plot the first sample with a different color and size
    plt.scatter(np.degrees(azi[0]), np.degrees(ele[0]), s=mag[0] * 300, c='black', marker='*')

    # Plot locs array
    locs_r = np.linalg.norm(locs, axis=1)  # radial distance
    locs_ele = np.nan_to_num(np.arccos(locs[:, 2] / locs_r)) # Elevation angle (theta)
    locs_azi = np.arctan2(locs[:, 1], locs[:, 0])  # azimuthal angle (phi)
    plt.scatter(np.degrees(locs_azi), np.degrees(locs_ele), s=100, c='black', marker='o')

    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Elevation (degrees)')
    plt.title('DOA Coordinates')
    plt.show()


def polar_plot_full(DOA_full, ir_full, num_samples):
    DOA = DOA_full[:num_samples]
    ir_full = ir_full.reshape(-1, 1)
    ir = ir_full[:num_samples]
    r = np.linalg.norm(DOA[:num_samples,:], axis=1)  # radial distance
    azi = np.arctan2(DOA[:num_samples, 1], DOA[:num_samples, 0])  # azimuthal angle (phi)
    mag = ir / np.max(ir)  # magnitude of each coordinate

    # Filter out DOA coordinates with non-zero elevation
    horiz_DOA = DOA[DOA[:num_samples, 2] == 0]
    horiz_azi = np.arctan2(horiz_DOA[:num_samples, 1], horiz_DOA[:num_samples, 0])
    horiz_mag = ir[DOA[:num_samples, 2] == 0].flatten() / np.max(ir)

    # Create polar plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(azi, r, s=mag * 100, c=mag, cmap='coolwarm', alpha=0.5)
    ax.scatter(horiz_azi, np.ones_like(horiz_azi) * np.max(r), s=horiz_mag * 100, c=horiz_mag, cmap='coolwarm',
               alpha=0.5)
    ax.set_rticks(np.arange(0, np.max(r) + 0.1, 0.1))
    ax.set_rlabel_position(22.5)
    # ax.grid(True)
    plt.title('Polar Plot of DOA')
    plt.show()


def plot_sphere(DOA, ir):
    r = np.linalg.norm(DOA, axis=1)  # radial distance
    ele = np.nan_to_num(np.arccos(DOA[:, 2] / r)) # Elevation angle (theta)
    azi = np.arctan2(DOA[:, 1], DOA[:, 0])  # azimuthal angle (phi)
    mag = ir / np.max(ir)  # magnitude of each coordinate

    # Convert spherical coordinates to cartesian coordinates
    x = r * np.sin(ele) * np.cos(azi)
    y = r * np.sin(ele) * np.sin(azi)
    z = r * np.cos(ele)

    # Plot 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=mag*100, c=mag, cmap='coolwarm', alpha=0.5)

    # Set axis limits and labels
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('DOA Coordinates on Unit Sphere')
    plt.show()


def plot_spherical_coordinates(cartesian_coords):
    """Plot spherical coordinates on a 2D scatter plot.

    Parameters
    ----------
    cartesian_coords : array_like
        Cartesian coordinates as a n x 3 numpy array.

    Returns
    -------
    None
    """
    # Convert cartesian coordinates to spherical coordinates
    r = np.linalg.norm(cartesian_coords, axis=1)  # radial distance
    theta = np.arccos(cartesian_coords[:, 2] / r)  # polar angle (theta)
    phi = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0])  # azimuthal angle (phi)

    # Convert spherical coordinates to degrees
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    # Create scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(phi, theta, r, c='b', marker='o')  # Customize color and marker as needed
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Elevation')
    ax.set_zlabel('Distance')
    ax.view_init(90, 0, 0)
    plt.show()


def plot_recorded_signal(sigs, fs):
    """Plot a time signal recorded using sounddevice.playrec()"""
    # print("Shape of sigs: ", sigs.shape)
    # Create a time vector
    time = np.arange(0, len(sigs)) / fs

    # Create a plot for the time signal
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axs[0].plot(time, sigs)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Recorded Signal')

    # Create a plot for the spectrogram
    f, t, Sxx = signal.spectrogram(sigs, fs)
    axs[1].pcolormesh(t, f, np.log10(Sxx), cmap='jet')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_title('Spectrogram')

    plt.show()


def plot_time_signals(signals, labels=None):
    num_signals = len(signals)

    if labels is not None and len(labels) != num_signals:
        raise ValueError("The number of labels should match the number of signals.")

    # Generate x-axis values
    x = np.arange(len(signals[0]))

    # Plot each signal
    for i, signal in enumerate(signals):
        label = labels[i] if labels is not None else None
        plt.plot(x, signal, label=label)

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def doa_time(doa):
    # Create subplots
    azimuth = np.arctan2(doa[:, 1], doa[:, 0])
    elevation = np.arctan2(doa[:, 2], np.sqrt(doa[:, 0] ** 2 + doa[:, 1] ** 2))

    fig, (ax1, ax2) = plt.subplots(3, 1, sharex=True)

    # Middle subplot for azimuth
    ax2.plot(azimuth)
    ax2.set_ylabel('Azimuth')
    ax2.set_title('Azimuth vs Time')
    ax2.set_ylim([-np.pi/4, np.pi/4])  # Set y-axis limits to -pi to pi
    ax2.set_xlim([0, 200])
    ax2.set_yticks([-np.pi, 0, np.pi])  # Set y-axis ticks to -pi, 0, pi
    ax2.set_yticklabels(['$-\pi$', '0', '$\pi$'])  # Set y-axis tick labels to -pi, 0, pi

    # Bottom subplot for elevation
    ax1.plot(elevation)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Elevation')
    ax1.set_title('Elevation vs Time')
    ax1.set_ylim([-np.pi / 2, np.pi / 2])  # Set y-axis limits to -pi/2 to pi/2
    ax1.set_xlim([0, 200])
    ax1.set_yticks([-np.pi / 2, 0, np.pi / 2])  # Set y-axis ticks to -pi/2, 0, pi/2
    ax1.set_yticklabels(['$-\pi/2$', '0', '$\pi/2$'])  # Set y-axis tick labels to -pi/2, 0, pi/2

    # Set shared x-axis label
    # fig.text(0.5, 0.04, 'Time', ha='center')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Show plot
    plt.show()


def doa_time2(doa, doa2, doa3, N, N2, ir):
    # Create subplots
    azimuth = np.arctan2(doa[:, 1], doa[:, 0])
    elevation = np.arctan2(doa[:, 2], np.sqrt(doa[:, 0] ** 2 + doa[:, 1] ** 2))

    azimuth2 = np.arctan2(doa2[:, 1], doa2[:, 0])
    elevation2 = np.arctan2(doa2[:, 2], np.sqrt(doa2[:, 0] ** 2 + doa2[:, 1] ** 2))

    azimuth3 = np.arctan2(doa3[:, 1], doa3[:, 0])
    elevation3 = np.arctan2(doa3[:, 2], np.sqrt(doa3[:, 0] ** 2 + doa3[:, 1] ** 2))

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # Top subplot for IR (Impulse Response)
    ax0.plot(ir)
    ax0.set_ylabel('Amplitude')
    ax0.set_title('Impulse Response')
    ax0.set_xlim([0, 600])

    # Middle subplot for azimuth
    ax1.plot(azimuth, label='Raw DOA')
    ax1.plot(azimuth2, label=f'K = {N}')
    ax1.plot(azimuth3, label=f'K = {N2}')
    ax1.set_ylabel('Azimuth in deg')
    ax1.set_title('Azimuth vs Time')
    ax1.set_ylim([-np.pi/4, np.pi/4])  # Set y-axis limits to -pi to pi
    ax1.set_xlim([0, 600])
    ax1.set_yticks([-np.pi, 0, np.pi])  # Set y-axis ticks to -pi, 0, pi
    ax1.set_yticklabels(['$-180$', '0', '$180$'])  # Set y-axis tick labels to -pi, 0, pi
    ax1.legend(loc="lower left", prop={'size': 10})  # Add legend

    # Bottom subplot for elevation
    ax2.plot(elevation, label='Raw DOA')
    ax2.plot(elevation2, label=f'K = {N}')
    ax2.plot(elevation3, label=f'K = {N2}')
    ax2.set_xlabel('Time (samples)')
    ax2.set_ylabel('Elevation in deg')
    ax2.set_title('Elevation vs Time')
    ax2.set_ylim([-np.pi/2, np.pi/2])  # Set y-axis limits to -pi/2 to pi/2
    ax2.set_xlim([0, 600])
    ax2.set_yticks([-np.pi/2, 0, np.pi/2])  # Set y-axis ticks to -pi/2, 0, pi/2
    ax2.set_yticklabels(['$-90$', '0', '$90$'])  # Set y-axis tick labels to -pi/2, 0, pi/2
    ax2.legend(loc="lower left", prop={'size': 10})  # Add legend

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Show plot
    plt.show()


def plot_coordinates_on_sphere(coordinates, image_path, time_values, scalar_vector):
    # Load the image of the head
    head_image = plt.imread(image_path)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the head image as a texture
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=head_image, rstride=4, cstride=4)

    # Scale the axes equally to form a sphere
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    # Compute the magnitude of the scalar vector
    scalar_magnitude = np.linalg.norm(scalar_vector)

    # Normalize the scalar vector
    normalized_scalar = scalar_vector / scalar_magnitude

    # Plot the coordinates on the unit sphere with scaled arrow sizes and time-based color
    for i in range(coordinates.shape[1]):
        ax.quiver(
            0, 0, 0,
            coordinates[0, i], coordinates[1, i], coordinates[2, i],
            color=cm.viridis(time_values[i]), alpha=0.8,
            arrow_length_ratio=0.05 * np.linalg.norm(coordinates[:, i]),
            linewidth=0.5,
            pivot='tail'
        )

    # Create a colorbar for the time values
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(time_values)
    cbar = plt.colorbar(sm)
    cbar.set_label('Time')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Coordinates on Unit Sphere')

    # Show the plot
    plt.show()


def plot_spectrogram(ir, fs,title):
    # Calculate the spectrogram
    f, t, Sxx = signal.spectrogram(ir, fs)

    # Plot the spectrogram
    plt.pcolormesh(t, f, np.log10(Sxx))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.ylim([20, 20000])
    plt.colorbar()
    plt.show()


def plot_time_signals(signals, xlabel='Time', ylabel='Amplitude', title='Time Signals'):
    num_signals = signals.shape[1]
    time = np.arange(signals.shape[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    for i in range(num_signals):
        ax1.plot(time, signals[:, i], label=f'Signal {i + 1}')
        ax2.plot(time, 20 * np.log10(np.abs(signals[:, i])), label=f'Signal {i + 1}')

    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Magnitude (dB)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_spectrograms(signals):
    num_plots = 6
    num_rows = 2
    num_cols = 3

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)

    for i in range(num_plots):
        ax = axes[i // num_cols, i % num_cols]
        ax.specgram(signals[:, i], Fs=2)  # Adjust Fs if needed
        ax.set_title(f"Spectrogram {i + 1}")

    plt.show()


def plot_ir_and_spectrum(ir, fs, number=1):
    ir = ir - np.mean(ir)
    # Plot impulse response and frequency spectrum as subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    # Plot impulse response
    ax1.plot(ir)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Impulse Response')

    # Compute frequency spectrum
    n_fft = len(ir)
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    spectrum = 20*np.log10(np.abs(np.fft.rfft(ir)))

    # Plot frequency spectrum
    ax2.plot(freqs, spectrum, 'r')
    ax2.set_ylabel('Magnitude')
    ax2.set_ylim(bottom=-40)
    ax2.set_xlim(left=20, right= fs /2)
    ax2.set_xscale('log')
    ax2.set_title('Frequency Spectrum' + str(number))

    plt.tight_layout()
    plt.show()


def plot_DOA_Sph(coords):
    """
    Plots spherical directions of arrival on a unit sphere

    Args:
    coords (numpy array): nx3 array of Cartesian coordinates

    Returns:
    None
    """

    # Convert Cartesian to spherical coordinates
    r = np.linalg.norm(coords, axis=1)  # radial distance
    theta = np.arccos(coords[:, 2] / r)  # polar angle (theta)
    phi = np.arctan2(coords[:, 1], coords[:, 0])  # azimuthal angle (phi)

    # Convert to degrees
    elev, azim = np.rad2deg(theta), np.rad2deg(phi)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(azim, elev, s=50 * r, alpha=0.5)
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_zlim([-90, 90])
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    plt.plot(azim)
    plt.xlabel('Time')
    plt.ylabel('Azimuth')
    plt.show()

    plt.plot(elev)
    plt.xlabel('Time')
    plt.ylabel('Elevation')
    plt.show()