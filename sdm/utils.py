from scipy.signal.windows import cosine
from spaudiopy import utils, decoder, sph, decoder, process, grids
import sounddevice as sd
from scipy.io import wavfile
import numpy as np
import math
import sys
import soundfile as sf


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy import signal, io


def distances(p1, p2):
    """Calculates the distance in 3D space between two points"""
    return np.round(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2))


def window_check(win_len, max_len, fs):
    """ Check if the window length is properly set. If the window size is smaller than
    the time it takes for soundto travel between the two farthest microphones
    it has to be reconfigured.
    Condition: window size (samples) > TravelTimeFarthest (samples)

    Parameters
    ----------
    winLen = Window Length in samples
    maxLen = The furthest distance between two microphones
    fs = Sampling Frequency

    Returns
    -------
    bool(check)
    """
    dt = 1 / fs
    c = 343
    # CONDITION: WINDOW LENGTH > T_Travel
    win_len_sec = dt * win_len
    t_travel = 2 * (max_len / c)

    if t_travel <= win_len_sec:
        print('Window Length is fine')
        check = bool(True)
    else:
        print('Longer window length is needed')
        check = bool(False)
    return check


def sph2cart(doa_sph):
    """
    Spherical To Cartesian Coordinates
    :param doa_sph: nx3 array with azimuth, elevation, and distance
    :return: nx3 array with x, y, and z coordinates
    """
    azi = (doa_sph[:, 0])
    colat = (doa_sph[:, 1])
    r = doa_sph[:, 2]

    x = r * np.sin(colat) * np.cos(azi)
    y = r * np.sin(colat) * np.sin(azi)
    z = r * np.cos(colat)
    doa_cart = np.vstack((x, y, z)).T
    return doa_cart


def cart2sph(coords, type='degrees'):
    """
    Convert coordinates from Cartesian to spherical.

    Args:
        coords (numpy array): Array of shape (n, 3) with x, y, z coordinates.

    Returns:
        Tuple[numpy array, numpy array]: A tuple containing two arrays of shape (n, 3),
        one with spherical coordinates in radians (ele, azi, r), and the other with
        spherical coordinates in degrees (ele_deg, azi_deg, r).
    """

    if len(coords.shape) == 1:
        coords = np.array([coords])

    # Suppress/hide the warning
    np.seterr(invalid='ignore')
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ele = np.arccos(z / r)
    azi = np.arctan2(y, x)
    ele_deg = np.degrees(ele)
    azi_deg = np.degrees(azi)
    np.nan_to_num(ele, copy=True, nan=0.0, posinf=None, neginf=None)
    np.nan_to_num(ele_deg, copy=True, nan=0.0, posinf=None, neginf=None)
    doa_sph_rad = np.column_stack((azi, ele, r))
    doa_sph_deg = np.column_stack((azi_deg, ele_deg, r))
    doa_sph_rad[np.isnan(doa_sph_rad)] = 0
    doa_sph_deg[np.isnan(doa_sph_deg)] = 0

    return doa_sph_rad, doa_sph_deg


def rotate_cartesian(coords, angle):
    """
    Rotates cartesian coordinates by an angle in degrees
    """
    angle_rad = np.deg2rad(angle)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                           [np.sin(angle_rad), np.cos(angle_rad), 0],
                           [0, 0, 1]])
    return np.dot(rot_matrix, coords.T).T


def asarray_1d(a, **kwargs):
    """Squeeze the input and check if the result is one-dimensional.
    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.
    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def deg2rad(deg):
    """Convert from degree [0, 360) to radiant [0, 2*pi)."""
    return deg % 360 / 180 * np.pi


def rad2deg(rad):
    """Convert from radiant [0, 2*pi) to degree [0, 360)."""
    return rad / np.pi * 180 % 360


def find_max_and_set_zero(array, num_indices=10):
    max_values = np.max(array, axis=0)

    for i in range(array.shape[1]):
        max_index = np.argmax(array[:, i])
        start_index = max_index
        end_index = min(max_index + num_indices, array.shape[0])
        array[:start_index, i] = 0
        array[end_index:, i] = 0

    return array


def write_to_wav(filename, data, samplerate):
    """
    Write each row of a 2D NumPy array as a separate channel in a WAV file.

    Args:
        filename (str): The output file path.
        data (ndarray): 2D NumPy array containing the audio data.
        samplerate (int): The sample rate of the audio data.

    Returns:
        None
    """
    # Ensure data is a 2D array
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")

    # Get the number of channels and frames
    num_channels, num_frames = data.shape

    # Transpose the data to have channels along the first axis
    transposed_data = np.transpose(data)

    # Scale the data to the range of the desired integer data type
    scaled_data = (transposed_data * 32767).astype(np.int16)

    # Write the WAV file
    wavfile.write(filename, samplerate, scaled_data)


def compare_arrays(arr1, arr2):
    # Element-wise comparison
    comparison = arr1 == arr2

    # Count the number of matches
    num_matches = np.count_nonzero(comparison)

    # Check if arrays are equal
    are_equal = np.all(comparison)

    return are_equal, num_matches


def estimate_distance(ir, fs):
    """
    Estimate the distance between a source and a microphone based on an impulse response and sample frequency.

    Args:
        ir (numpy array): Impulse response of the system.
        fs (int): Sample frequency in Hz.

    Returns:
        float: Estimated distance between the source and microphone in meters.
    """
    speed_of_sound = 343  # Speed of sound in meters per second

    # Find the peak of the impulse response
    peak_index = np.argmax(ir)

    # Calculate the time delay of the peak relative to the start of the impulse response
    time_delay = peak_index / fs

    # Calculate the estimated distance using the time delay and the speed of sound
    distance = time_delay * speed_of_sound

    print(f'Distance = {distance} m')



def normalize(impulse_responses):
    # Find the maximum value across all impulse responses
    max_value = np.max(impulse_responses)
    print(max_value)
    # Normalize the impulse responses in the third dimension
    normalized_impulse_responses = impulse_responses / max_value

    return normalized_impulse_responses


def write_SSR_IRs(filename, time_data_l, time_data_r, wavformat="float32", fs=48000):
    """Takes two time signals and writes out the horizontal plane as HRIRs for
    the SoundScapeRenderer. Ideally, both hold 360 IRs but smaller sets are
    tried to be scaled up using repeat.
    Parameters
    ----------
    filename : string
        filename to write to
    time_data_l, time_data_r : io.ArraySignal
        ArraySignals for left/right ear
    wavformat : {float32, int32, int16}, optional
        wav file format to write [Default: float32]
    Raises
    ------
    ValueError
        in case unknown wavformat is provided
    ValueError
        in case integer format should be exported and amplitude exceeds 1.0
    """
    # make lower case and remove spaces
    wavformat = wavformat.lower().strip()

    # equator_IDX_left = utils.nearest_to_value_logical_IDX(
    #     time_data_l.grid.colatitude, _np.pi / 2
    # )
    # equator_IDX_right = utils.nearest_to_value_logical_IDX(
    #     time_data_r.grid.colatitude, _np.pi / 2
    # )

    # irs_left = time_data_l.signal.signal[equator_IDX_left]
    # irs_right = time_data_r.signal.signal[equator_IDX_right]
    irs_left = time_data_l# .signal.signal
    print("Shape of irs_left: ", time_data_l.shape)
    irs_right = time_data_r# .signal.signal

    irs_to_write = interleave_channels(
        left_channel=irs_left, right_channel=irs_right, style="SSR"
    )
    # data_to_write = utils.simple_resample(
    #     irs_to_write, original_fs=time_data_l.signal.fs, target_fs=44100
    # )
    data_to_write = irs_to_write

    # get absolute max value
    max_val = np.abs([time_data_l, time_data_r]).max()
    if max_val > 1.0:
        if "int" in wavformat:
            raise ValueError(
                "At least one amplitude value exceeds 1.0, exporting to an "
                "integer format will lead to clipping. Choose wavformat "
                '"float32" instead or normalize data!'
            )
        print("WARNING: At least one amplitude value exceeds 1.0!", file=sys.stderr)

    if wavformat in ["float32", "float"]:
        io.wavfile.write(
            filename=filename,
            rate=fs,
            data=data_to_write.T.astype(np.float32),
        )
    elif wavformat in ["int32", "int16"]:
        io.wavfile.write(
            filename=filename,
            rate=fs,
            data=(data_to_write.T * np.iinfo(wavformat).max).astype(wavformat),
        )
    else:
        raise ValueError(
            f'Format "{wavformat}" unknown, should be either "float32", "int32" or "int16".'
        )


def interleave_channels(left_channel, right_channel, style=None):
    """Interleave left and right channels. Style == 'SSR' checks if we total
    360 channels.
    """
    if not left_channel.shape == right_channel.shape:
        raise ValueError(
            "left_channel and right_channel have to be of same dimensions!"
        )

    if style == "SSR" and not (left_channel.shape[0] == 360):
        raise ValueError("Provided arrays to have 360 channels (Nchannel x Nsamples).")

    output_data = np.repeat(left_channel, 2, axis=0)
    output_data[1::2, :] = right_channel

    return output_data


def load_mic(filename):
    mic_locs_filename = filename
    mic_locs_dict = io.loadmat(mic_locs_filename)
    mic_locs = mic_locs_dict['micLocs']  # Numpy array
    return mic_locs


def resample(audio_filename, target_fs):
    [audio_in, fs_x] = sf.read(audio_filename)
    if audio_in.ndim > 1:
        # Convert stereo to mono
        audio_in = np.mean(audio_in, axis=1)
    if fs_x != target_fs:
        resample_ratio = float(target_fs) / fs_x
        audio_in = signal.resample(audio_in, int(len(audio_in) * resample_ratio))

    return audio_in