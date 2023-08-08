from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lna
import time
from scipy.signal.windows import cosine
import scipy.signal as sps
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.signal import butter, filtfilt
from scipy.io import savemat
from scipy.fft import ifft
from sdm import utils
import math


def sdm_par(IR, p):
    """
    Spatial Decomposition Method, implemented as parallel processing
    Returns locations (or direction of arrival) at each time step of the IR.

    Adapted from MATLAB's SDM Toolbox: https://se.mathworks.com/matlabcentral/fileexchange/56663-sdm-toolbox
    Original author: Sakari Tervo

    :param IR: A matrix of measured impulse responses from a microphone array [N numberOfMics]
    :param p: Parameters can be given as a dict:
              Radius : float : Radius of the microphones [m]
              mic_locs : numpy array : Microphone positions in Cartesian coordinates [m]
              fs : float : sampling frequency [Hz]
              c : float : speed of sound [m/s]
              winLen : int : length of the processing window, if empty, minimum size is used.
              parFrames : int : parallel frames in the processing, maximum is length(IR), and minimum is 1.
    :return: DOA : 3-D Cartesian locations of the image-sources [N 3];

    p = createSDMStruct('mic_locs', mic_locs,'c', c, ...'fs',fs,'parFrames',a_number,'winLen',b_number)
    """
    print('Started SDM processing')
    start_time = time.time()
    print(np.argmax(IR, axis=0))
    # Constants for array processing
    num_mics = p['mic_locs'].shape[0]  # number of microphones
    pairs = np.array(list(combinations(range(num_mics), 2)))  # microphone pairs
    num_pairs = pairs.shape[0]  # number of microphone pairs
    V = p['mic_locs'][pairs[:, 0], :] - p['mic_locs'][pairs[:, 1], :]  # microphone vector difference matrix
    D = np.sqrt(np.sum(V ** 2, axis=1))

    # Choose a frame size
    # win_len_min = 2*np.ceil(np.max(D)/p['c'])
    win_len_min = 2 * np.ceil(np.max(D) / (p['c'] * p['fs'])) + 8
    print(f"win_len_min: {win_len_min}")
    if 'win_len' not in p or p['win_len'] < win_len_min:
        win_len = win_len_min
        print("Using frame size" + str(win_len))
    else:
        win_len = p['win_len']

    # Windowing
    window = sps.windows.hann(win_len)
    print(IR[0, 0])
    max_td = np.ceil(np.max(D)/p['c']*p['fs'])+1  # Maximum accepted time delay difference
    max_td = int(max_td)
    inv_V = lna.pinv(V)
    offset = int(win_len/2+1)
    # offset_v = offset+np.arange(-max_td, max_td+1).astype(int)  # Region of interest in TDOA estimation
    offset_v = np.array([offset + i for i in range(-max_td, max_td + 2)]) - 1

    eps = 2.2204e-16
    # Variables for the frame based processing in the for loop
    startx = 0
    nforward = 1
    endx = startx + win_len - 1
    num_frames = int(np.floor((len(IR) - win_len) // nforward))

    # This indexing is required for the parallel processing
    base_inds = np.arange(startx, endx + 1)
    base_inds = base_inds.T
    base_par_inds = np.tile(base_inds[:, np.newaxis], (1, p['par_frames']))
    for nn in range(p['par_frames']):
        base_par_inds[:, nn] += nforward * nn

    # Also this is for the parallel processing
    par_pairs = np.zeros((num_pairs * p['par_frames'], 2))
    for nn in range(1, p['par_frames'] + 1):
        par_pairs[(nn - 1) * num_pairs: nn * num_pairs, :] = pairs + (nn - 1) * num_mics

    # MAIN PROCESSING LOOP
    DOA = np.zeros((len(IR), 3))
    for n in range(1, num_frames, int(p['par_frames'])):
        # Select the frames that are to be processed
        cur_frames = list(range(n, min(n + p['par_frames'], num_frames)))
        n_cur_frames = max(np.shape(cur_frames))
        n_cur_frames = int(n_cur_frames)
        # Create overlapping indexes
        par_inds = base_par_inds[:, :n_cur_frames] + n - 1

        win_len = int(win_len)
        n_cur_frames = int(n_cur_frames)
        num_mics = int(num_mics)

        par_inds = np.round(par_inds).astype(int)
        # Index time-domain signal into parallel overlapping windows
        temp = IR[np.ravel(par_inds, 'F'), :]
        if temp.shape != (win_len, n_cur_frames, num_mics):
            temp = np.reshape(temp, (win_len, n_cur_frames, num_mics), order='F')
        temp = np.transpose(temp, (0, 2, 1))
        temp = np.reshape(np.transpose(temp, (0, 2, 1)), (win_len, num_mics * n_cur_frames))

        # Apply windowing
        par_samples = window[:, np.newaxis] * temp

        # Run parallel fft for overlapping windows
        X = np.fft.fft(par_samples, axis=0)

        X1 = X[:, par_pairs[0:n_cur_frames * num_pairs, 0].astype(int)]
        X2 = X[:, par_pairs[0:n_cur_frames * num_pairs, 1].astype(int)]

        # Cross Power Spectrum between microphone pairs
        P12 = np.multiply(X1, np.conj(X2))

        # Cross-Correlation between microphone pairs
        Rtemp = np.real(ifft(P12, axis=0))

        R = fftshift22(Rtemp)

        eps = np.finfo(float).eps  # Get machine epsilon

        R = np.where(R < (eps * 2), eps * 2, R)
        # Remove time delays that are beyond [-t_max, t_max]
        R2 = R[offset_v, :]
        # Find maximum
        indk = np.argmax(R2, axis=0)

        # Compensate the offset
        indk = indk + offset - (max_td + 1)

        # Interpolate the maximum peak by assuming exponential shape
        tau = interpolate_tau(R, indk, True)
        tau_mat = np.reshape(tau, [num_pairs, n_cur_frames], order='F')
        d = tau_mat / p['fs']
        # Solve the direction of arrival.d = tau_mat / p.fs
        #
        # # Solve the direction of arrival
        # # Assuming inv_V is the inverse of matrix V
        k = np.matmul(-inv_V, d)

        # Normalize k to unity to obtain DOA
        k = k * (1. / np.sqrt(np.sum(k ** 2, axis=0)))
        cur_frames = np.array(cur_frames)

        fs = float(p['fs'])
        c = float(p['c'])
        # The distance that the sound has traveled
        d_n = (cur_frames + int(win_len) / 2) / fs * c

        # Save the 'Image Sources', i.e, locations of the reflections
        DOA[cur_frames + int(win_len) // 2, :] = np.multiply(k.T, d_n[:, np.newaxis])
        # --- EOF Frame based processing ---

    DOA = np.nan_to_num(DOA, nan=0)
    print('Ended SDM processing in', time.time() - start_time, 'seconds.')
    return DOA


def fftshift(x):
    n = x.shape[0]
    i = np.concatenate((np.arange(n//2+1, n+1), np.arange(1, n//2+1)))
    print(i)
    y = x[i-1, :]
    return y

def fftshift22(x):
    n = x.shape[0]
    shift = n // 2
    y = np.roll(x, shift, axis=0)
    return y


def interpolate_tau(R, ind, offset_flag):
    """
    Adapted from Matlab SDM Toolbox
    :param R: cross correlation matrix, it is required that R > 0, for all ind +- 1
    :param ind: maximum indices in the vector
    :param offset_flag: if true, set the time delay to length(R)/2 + 1
    :return: tau
    """

    offset = 0
    if offset_flag:
        offset = np.floor(R.shape[0] / 2)

    # index to time difference of arrival conversion
    t = ind
    t2 = t - offset

    # Time indices for all the rows in the matrix
    inds_nom = R.shape[1-1] * (np.arange(0, (R.shape[2-1] - 1)+1)) + t
    R = R.reshape((1, R.shape[0] * R.shape[1]), order="F")

    inds_nom = inds_nom.astype(int)

    # Solve the time delay from [-1 0 1] samples around the maximum
    c = (np.log(R[0, inds_nom + 1]) - np.log(R[0, inds_nom - 1])) / \
        (4 * np.log(R[0, inds_nom]) - 2 * np.log(R[0, inds_nom - 1]) - 2 * np.log(R[0, inds_nom + 1]))

    # Add the initial integer delay to the interpolated delay
    tau = t2 + c
    return tau


def align_DOA(IR, DOAs_cartesian, source_position, tolerance=5):
    # Find the index of the maximum value in the impulse response
    max_index = np.argmax(IR)

    # Calculate the angle between the DOA at the maximum index and the source position
    angle = math.degrees(math.acos(np.dot(DOAs_cartesian[max_index], source_position) / (
                np.linalg.norm(DOAs_cartesian[max_index]) * np.linalg.norm(source_position))))

#    if angle <= tolerance:
#        # The DOA at the maximum index is within the tolerance range, no rotation needed
#        aligned = False
#        return DOAs_cartesian
#    else:
    # Calculate the rotation angle to align the DOA at the maximum index with the source position
    rotation_angle = angle - tolerance

    # Rotate the DOAs by the negative rotation angle
    DOAs_cartesian = utils.rotate_cartesian(DOAs_cartesian, rotation_angle)
    aligned = True

    return DOAs_cartesian, aligned


def align_DOA2(IR, DOAs_cartesian, angle, tolerance=5):
    # Find the index of the maximum value in the impulse response
    max_index = np.argmax(IR)

#    if angle <= tolerance:
#        # The DOA at the maximum index is within the tolerance range, no rotation needed
#        aligned = False
#        return DOAs_cartesian
#    else:

    # Rotate the DOAs by the negative rotation angle
    DOAs_cartesian = utils.rotate_cartesian(DOAs_cartesian, angle=angle)
    aligned = True

    return DOAs_cartesian, aligned


def align_ir(ir_omni, ir_spatial):
    # Calculate the maximum indices for ir_omni and ir_spatial
    max_index_omni = np.argmax(ir_omni)
    max_index_spatial = np.argmax(ir_spatial, axis=1)
    max_idx_spatial = np.mean(max_index_spatial)
    # Calculate the alignment offset
    offset = max_index_omni - max_idx_spatial
    offset = int(np.floor(offset))
    print("offset: ", offset)
    # Adjust the alignment of ir_spatial using the offset
    aligned_ir_spatial = np.roll(ir_spatial, offset, axis=1)

    return aligned_ir_spatial




def bp_filt(data, fs, lowcut, highcut, order=4):
    # Define the filter parameters
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    # Apply the bandpass filter to each column of the data
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        b, a = butter(order, [low, high], btype='band')
        filtered_data[:, i] = filtfilt(b, a, data[:, i])

    return filtered_data


def plot_array(mic_locs):
    # Plot the geometry of microphone, defined in mic_locs

    numOfMics = mic_locs.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for m in range(numOfMics):
        ax.plot([0, mic_locs[m, 0]], [0, mic_locs[m, 1]], [0, mic_locs[m, 2]], color='k')
        ax.scatter(mic_locs[m, 0], mic_locs[m, 1], mic_locs[m, 2], color='k', marker='o')
        ax.text(mic_locs[m, 0], mic_locs[m, 1], mic_locs[m, 2], f"mic #{m + 1}")
    ax.grid(True)
    ax.view_init(elev=35, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Microphone Array Geometry in the Analysis')
    ax.set_xlabel('X-coord.')
    ax.set_ylabel('Y-coord.')
    ax.set_zlabel('Z-coord.')
    plt.show()


def running_mean(data, N):
    """
    Calculates the moving average of a numpy array.

    Args:
    :param data : The input data.(numpy.ndarray)
    :param N : The size of the moving average window.  (int)

    Returns:
    :return: An array containing the moving average values.
    """
    kernel = np.ones(N) / N
    return np.convolve(data, kernel, mode='same')


def create_sdm_dict(mic_locs=None, fs=48000, c=345, win_len=0, par_frames=8192, show_array=False):
    """
    Creates a structure of parameters for SDM_par function.

    Parameters:
    -----------
    :param mic_locs : array_like, shape (n_mics, 3), optional
        User defined microphone locations. The origin is at the center of the array.
        They should be in the same order as the impulse responses in the IR matrix for SDMpar.
        However, it is also important that the correct sampling frequency is set.
    :param fs : float, optional
    :param c : float, optional
        Speed of sound in m/s. Default is 345.
    :param win_len : int, optional
        Window length in samples. Default is 0, which sets the minimum window length.
    :param par_frames : int, optional
        Number of parallel frames in SDMpar. Default is 8192.
   :param show_array : bool, optional
        Display the microphone array in a figure. Default is False.

    Returns:
    --------
    :return:p : dict
        A dictionary of parameter values.

    Examples:
    """
    # Check if all arguments are valid
    valid_names = ['mic_locs', 'fs', 'c', 'win_len', 'par_frames', 'show_array']
    for i in range(0, len(valid_names), 2):
        if valid_names[i] not in locals() and valid_names[i] not in globals():
            raise ValueError(f"Unknown parameter {valid_names[i]}")

    # Default values
    values = [fs, c, win_len, par_frames, show_array]

    for i in range(len(valid_names) - 1):
        if valid_names[i] in locals() or valid_names[i] in globals():
            values[i] = locals().get(valid_names[i], globals().get(valid_names[i]))

    p = {valid_names[i]: values[i] for i in range(len(valid_names) - 1)}

    # Custom microphone array
    if mic_locs is not None:
        if not isinstance(mic_locs, np.ndarray):
            raise TypeError("MicLocs must be a NumPy ndarray")
        if mic_locs.shape[1] != 3:
            raise ValueError("MicLocs must be a matrix of shape (n_mics, 3)")
        p['mic_locs'] = mic_locs

    return p


def render_brir(hrtf_data, doa_data, ir_data):
    """
    Generate a binaural room impulse response (BRIR) using given HRTF, DOA, and IR data.
    :param hrtf_data: HRTF data (numpy array) with shape (N, 2), where N is the number of HRTF measurements,
                      and each row represents the left and right HRTF data for a given measurement
    :param doa_data: DOA data (3 x n numpy array) with Cartesian coordinates, where n is the number of DOA positions,
                    and each column represents the x, y, and z coordinates of a DOA position
    :param ir_data: IR data (numpy array) with shape (L,), where L is the length of the impulse response
    :return: brir_l, brir_r, two binaural room impulse responses. one for each output channel
    """
    ir_data = ir_data.copy().T

    positions = hrtf_data.Source.Position.get_values(system='cartesian')

    len_ir = int(len(ir_data))
    len_hrir = hrtf_data.Dimensions.N
    brir_l = np.zeros([len_ir+len_hrir - 1])
    brir_r = np.zeros([len_ir+len_hrir - 1])


    # Assuming doa_data has shape (3, n) and ir_data has shape (L,)
    pos_idx = find_closest_indices(doa_data, positions)
    curr_H = hrtf_data.Data.IR.get_values(indices={"M": pos_idx, "R": [0, 1], "E": 0}).T
    for i in range(len(ir_data)):
        # curr_H = hrtf_data.Data.IR.get_values(indices={"M": pos_idx[i], "R": [0, 1], "E": 0}).T # Current Left HRTF

        brir_l[i:i + len_hrir] += curr_H[:, 0, i] * ir_data[i]
        brir_r[i:i + len_hrir] += curr_H[:, 1, i] * ir_data[i]

    return brir_l, brir_r


def render_brir_anchor(hrtf_data, doa_data, ir_data):
    """
    Render an anchor BRIR with just one DOA
    :param hrtf_data:
    :param doa_data:
    :param ir_data:
    :return:
    """
    ir_data = ir_data.copy().T
    doa_data = doa_data.copy().T

    positions = hrtf_data.Source.Position.get_values(system='cartesian')

    len_ir = int(len(ir_data))
    len_hrir = hrtf_data.Dimensions.N

    # Find the closest DOA positions to the peak DOA
    pos_idx = find_closest_indices(doa_data, positions)

    hrir = hrtf_data.Data.IR.get_values(indices={"M": pos_idx, "R": [0, 1], "E": 0}).T

    brir_l = sps.convolve(ir_data, hrir[:, 0, 0], mode='full')
    brir_r = sps.convolve(ir_data, hrir[:, 1, 0], mode='full')

    return brir_l, brir_r


def find_closest_indices(DOA, Positions):
    n_DOA = DOA.shape[0]
    closest_indices = np.zeros(n_DOA, dtype=int)
    for i in range(n_DOA):
        dist = np.linalg.norm(DOA[i] - Positions, axis=1)
        closest_indices[i] = np.argmin(dist)
    return closest_indices


def fade_signal(sig, win_in_len=0, win_out_len=0):
    """
    Apply a fade-in and / or fade-out to the signal in the form of one-sided
    cosine squared windows.
    Args:
        sig (`numpy.ndarray`): ``(..., samples)`` shape time domain signal(s).
        win_in_len (`int`, optional): length of fade-in window applied to the
            signal in samples, default 0.
        win_out_len (`int`, optional): length of fade-out window applied to the
            signal in samples, default 0..
    Returns:
        `numpy.ndarray`: ``(..., samples)`` shape time domain signal(s).
    """
    if win_in_len:
        win_in_len = int(win_in_len)  # in samples
        win = cosine(M=win_in_len * 2, sym=True) ** 2
        sig[..., :win_in_len] *= win[:win_in_len]

    if win_out_len:
        win_out_len = int(win_out_len)  # in samples
        win = cosine(M=win_out_len * 2, sym=True) ** 2
        sig[..., -win_out_len:] *= win[win_out_len:]

    return sig


def deconvolve(exc_sig, rec_sig, fs=None, f_low=None, f_high=None, res_len=None,
               filter_args=None, deconv_type='lin', deconv_phase=True,
               inv_dyn_db=None, win_in_len=None, win_out_len=None):
    """Deconvolve signals.
    Perform signal deconvolution of recording input spectra over excitation
    output spectra.
    Arguments:
        exc_sig (`numpy.ndarray`): ``(..., n_ch, n_samp)`` shape excitation
            output signals.
        rec_sig (`numpy.ndarray`): ``(..., n_ch, n_samp)`` shape recorded
            input signals.
        fs (`float`, optional): sampling frequency in Hertz, default ``None``
        f_low (`float`, optional): lower bandpass (or highpass) cutoff
            frequency in Hertz (or in normalized frequency in case no
            sampling frequency is given), default ``None``.
        f_high (`float`, optional): upper bandpass (or lowpass cutoff
            frequency in Hertz (or in normalized frequency in case no
            sampling frequency is given), default ``None``.
        res_len (`float`, optional): target length of deconvolution results
            in seconds (or in samples in case no sampling frequency is given),
            default ``None`` resembling no truncation.
        filter_args (optional): arguments will be passed to
            `scipy.signal.iirfilter`.
        deconv_type (``'lin'`` or ``'cyc'``, optional): linear deconvolution to
             cut non-harmonic distortion products from the resulting signals
             or cyclic deconvolution otherwise, default ``'lin'``.
        deconv_phase (`bool`, optional): if the phase of the excitation signals
            should be considered (complex deconvolution) or neglected otherwise
            (compensation of the magnitude spectrum), default ``True``.
        inv_dyn_db (`float`, optional): inversion dynamic limitation applied to
            excitation signal in Decibel, default ``None``.
        win_in_len (`int`, optional): length of fade-in window applied to
            deconvolution results in samples, default ``None``.
        win_out_len (`int`, optional): length of fade-out window applied to
            deconvolution results in samples, default ``None``.
    Returns:
        `numpy.ndarray`: ``(..., n_ch, n_samp)`` shape deconvolved signals.
    """

    np.seterr(divide='ignore')

    exc_sig = np.atleast_2d(exc_sig)  # as [..., channels, samples]
    rec_sig = np.atleast_2d(rec_sig)  # as [..., channels, samples]
    if exc_sig.shape[-2] > 1 and exc_sig.shape[-2] != rec_sig.shape[-2]:
        raise ValueError(
            f'Mismatch of provided excitation output ({exc_sig.shape[-2]}) '
            f'vs. recorded input ({rec_sig.shape[-2]}) size.'
        )

    filter_args = filter_args or {}
    filter_args.setdefault('N', 8)
    if fs:
        filter_args.setdefault('fs', fs)
    if f_low not in (None, False) and f_high:
        filter_args.setdefault('Wn', (f_low, f_high))
        filter_args['btype'] = 'bandpass'
    elif f_low not in (None, False):
        filter_args.setdefault('Wn', f_low)
        filter_args['btype'] = 'highpass'
    elif f_high not in (None, False):
        filter_args.setdefault('Wn', f_high)
        filter_args['btype'] = 'lowpass'
    filter_args.setdefault('ftype', 'butter')
    filter_args['output'] = 'sos'

    # # Plot provided signals in time domain
    # import matplotlib.pyplot as plt
    # etc_rec = 20 * np.log10(np.abs(np.squeeze(rec_sig[0])))
    # etc_exc = 20 * np.log10(np.abs(np.squeeze(exc_sig[0])))
    # plt.plot(etc_rec.T, 'k', label='recorded')
    # plt.plot(etc_exc.T, 'r', label='excitation')
    # plt.title('Provided signals')
    # plt.xlabel('Samples')
    # plt.ylabel('Energy Time Curve in dB')
    # plt.legend(loc='best')
    # plt.grid(which='both', axis='both')
    # plt.xlim([0, etc_exc.shape[-1]])
    # plt.ylim([-120, 0]
    #          + np.ceil((np.nanmax(np.hstack((etc_exc, etc_rec))) / 5) + 1) * 5)
    # plt.tight_layout()
    # plt.show()

    # Determine processing length
    n_samples = max(rec_sig.shape[-1], exc_sig.shape[-1])
    if 'lin' in deconv_type.lower():
        n_samples *= 2
    elif 'cyc' not in deconv_type.lower():
        raise ValueError('Unknown deconvolution type `{}`'.format(deconv_type))

    # Transform into frequency domain including desired zero padding
    exc_fd_inv = 1. / np.fft.rfft(exc_sig, n=n_samples, axis=-1)
    rec_fd = np.fft.rfft(rec_sig, n=n_samples, axis=-1)

    # # Plot provided signals in frequency domain
    # import matplotlib.pyplot as plt
    f = np.fft.rfftfreq(n_samples, d=1. / fs)
    tf_rec = 20 * np.log10(np.abs(np.squeeze(rec_fd[0])))
    tf_exc = 20 * np.log10(np.abs(np.squeeze(
         np.fft.rfft(exc_sig[0], n=n_samples, axis=-1))))
    plt.semilogx(f, tf_rec.T, 'k', label='recorded')
    plt.semilogx(f, tf_exc.T, 'r', label='excitation')
    plt.title('Provided signals')
    plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
    plt.ylabel('Magnitude in dB')
    plt.legend(loc='best')
    plt.grid(which='both', axis='both')
    if fs is not None:
        plt.xlim([20, fs / 2])
    plt.ylim([-120, 0]
         + np.ceil((np.nanmax(np.hstack((tf_exc, tf_rec))) / 5) + 1) * 5)
    plt.tight_layout()
    plt.show()

    # Apply zero phase
    if not deconv_phase:
        exc_fd_inv = np.abs(exc_fd_inv)

    # Apply inversion dynamic limitation
    if inv_dyn_db not in (None, False):
        inv_dyn_min_lin = np.min(np.abs(exc_fd_inv)) * 10 ** (
                abs(inv_dyn_db) / 20)

        # Determine bins that need to be limited
        in_fd_where = np.abs(exc_fd_inv) > inv_dyn_min_lin

        # Substitute magnitude and leave phase untouched
        exc_fd_inv[..., in_fd_where] = (
                inv_dyn_min_lin
                * np.exp(1j * np.angle(exc_fd_inv[..., in_fd_where]))
        )

        # # Plot effect of dynamic limitation in frequency domain
        # import matplotlib.pyplot as plt
        # f = np.fft.rfftfreq(n_samples, d=1./fs)
        # tf_raw = 20 * np.log10(np.abs(np.squeeze(
        #         1. / np.fft.rfft(exc_sig[0], n=n_samples, axis=-1))))
        # tf_dyn = 20 * np.log10(np.abs(np.squeeze(exc_fd_inv[0])))
        # plt.semilogx(f, tf_raw.T, 'k', label='raw')
        # plt.semilogx(f, tf_dyn.T, 'r', label='limited')
        # plt.title('Effect of dynamic limitation')
        # plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
        # plt.ylabel('Magnitude in dB')
        # plt.legend(loc='best')
        # plt.grid(which='both', axis='both')
        # if fs is not None:
        #     plt.xlim(right=fs / 2)
        # plt.ylim([-120, 0]
        #          + np.ceil((np.nanmax(np.hstack((tf_raw, tf_dyn))) / 5) + 1) * 5)
        # plt.tight_layout()
        # plt.show()

    # Deconvolve signals
    res_tf = rec_fd * exc_fd_inv

    # Apply bandpass filter
    if 'Wn' in filter_args:
        filter_sos = sps.iirfilter(**filter_args)
        _, filter_tf = sps.sosfreqz(
            sos=filter_sos,
            worN=res_tf.shape[-1],
            whole=False,
        )
        res_tf *= filter_tf

        # # Plot effect of bandpass filter in frequency domain
        # import matplotlib.pyplot as plt
        # f = np.fft.rfftfreq(n_samples, d=1. / fs)
        # tf_raw = 20 * np.log10(np.abs(np.squeeze((res_tf / filter_tf)[0])))
        # tf_fil = 20 * np.log10(np.abs(np.squeeze(res_tf[0])))
        # plt.semilogx(f, tf_raw.T, 'k', label='raw')
        # plt.semilogx(f, tf_fil.T, 'r', label='filtered')
        # plt.title('Effect of bandpass filter')
        # plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
        # plt.ylabel('Magnitude in dB')
        # plt.legend(loc='best')
        # plt.grid(which='both', axis='both')
        # if fs is not None:
        #     plt.xlim([20, fs / 2])
        # plt.ylim([-120, 0]
        #          + np.ceil((np.nanmax(np.hstack((tf_raw, tf_fil))) / 5) + 1) * 5)
        # plt.tight_layout()
        # plt.show()

    # Determine result length
    if 'lin' in deconv_type.lower():
        n_samples /= 2
    if res_len is not None and res_len != 0:
        n_samples = res_len * (fs if fs is not None else 1)
    n_samples = int(n_samples)

    # Transform into time domain and truncate to target length
    res_ir = np.fft.irfft(res_tf, axis=-1)[..., :n_samples]
    # res_ir_copy = res_ir.copy()  # solely for plotting purposes

    # Apply start and end window
    res_ir = fade_signal(
        sig=res_ir,
        win_in_len=win_in_len,
        win_out_len=win_out_len,
    )

    #plot.plot_time_signals(res_ir[:, :500])
    # # Plot effect of result signals windowing in time domain
    # import matplotlib.pyplot as plt
    # etc_raw = 20 * np.log10(np.abs(np.squeeze(res_ir_copy[0])))
    # etc_win = 20 * np.log10(np.abs(np.squeeze(res_ir[0])))
    # plt.plot(etc_raw.T, 'k', label='raw')
    # if ((win_in_len is not None and win_in_len != 0)
    #         or (win_out_len is not None and win_out_len != 0)):
    #     plt.plot(etc_win.T, 'r', label='windowed')
    # plt.title('Result signals')
    # plt.xlabel('Samples')
    # plt.ylabel('Energy Time Curve in dB')
    # plt.legend(loc='best')
    # plt.grid(which='both', axis='both')
    # plt.xlim([0, res_ir.shape[-1]])
    # plt.ylim([-120, 0]
    #          + np.ceil((np.nanmax(np.hstack((etc_raw, etc_win))) / 5) + 1) * 5)
    # plt.tight_layout()
    # plt.show()
    #
    # # Plot result signals in frequency domain
    # import matplotlib.pyplot as plt
    f = np.fft.rfftfreq(n_samples, d=1. / fs)
    tf_res = 20 * np.log10(np.abs(np.squeeze(np.fft.rfft(res_ir, axis=-1))))
    plt.semilogx(f, tf_res.T)
    plt.title('Result signals')
    plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
    plt.ylabel('Magnitude in dB')
    plt.grid(which='both', axis='both')
    if fs is not None:
        plt.xlim([20, fs / 2])
    plt.ylim([-120, 0] + np.ceil((np.nanmax(tf_res) / 5) + 1) * 5)
    plt.tight_layout()
    plt.show()


    return res_ir


def apply_nai(audio, fs, gains, ls_setup):
    '''Apply non-allpass filters and gains to the input signal.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
        Input audio signal.
    fs : int
        Sampling frequency of the audio signal.
    gains : ndarray, shape (n_samples, n_loudspeakers)
        Matrix of loudspeaker gains.
    ls_setup : spaudiopy.layout.Layout
        Loudspeaker setup.

    Returns
    -------
    signals : list of ndarrays
        List of individual channel signals, with one element per loudspeaker.
    '''
    n_samples, n_ls = gains.shape
    signals = [np.zeros(n_samples) for _ in range(n_ls)]
    for i, signal in enumerate(signals):
        for j in range(n_ls):
            if gains[j, i] > 0:
                ir = ls_setup.get_loudspeaker_signal(j)
                signal += np.convolve(audio, ir, mode='same') * gains[j, i]
    return signals
