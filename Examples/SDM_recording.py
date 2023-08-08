"""
This script records or loads Impulse Responses to encode and decode and impulse response using SDM. Output can be set
to a static binaural room impulse response or a series of loudspeaker impulse responses for real time spatial audio.


Main Functions:
1. Recording or loading impulse responses using miniDSP MCHStreamer and MOTU Ultralite mk4.
2. Loading HRTF data from a SOFA file.
3. Generating and rendering Binaural Room Impulse Responses (BRIRs).
4. Rendering loudspeaker signals for a given audio input.
5. Optionally listening to the binaural or loudspeaker-rendered signals.

Descriptions to the sdm functions can be found in their respective
Note: This script uses external libraries such as sounddevice, numpy, scipy, spaudiopy, and others.
"""


import sounddevice as sd
import numpy as np
from scipy import signal as sig
from scipy.io import loadmat, savemat, wavfile
from sdm import utils, analysis, plot, capture
import time
import sofa as sofa
import spaudiopy as spa


start_time = time.time()

# Device Setup for miniDSP MCHStreamer and MOTU Ultralite mk4.
# Use OSX Audio/Midi Settings to verify channels for aggregate device
device_name = "Aggregate Device"                # Device name
device_id = 0                                   # Device ID according to printing
input_idx = [1, 3, 5, 7, 9, 11, 13, 17, 19]     # Input channel ID. Start from 1
output_idx = [1, 2, 3]                          # Output Channel ID. Start from 1

fs = 48000                                      # Desired Sampling Rate in Hz
num_channels = 34                               # Total number of channels with Aggregate Device

loopback_idx = 8                                # Index of loopback channel in Aggregate Device
num_sources = 1                                 # number of sources to record

# Configure the sine sweep
sweep_config = {
    'sweep_hp': 40,                             # Start frequency of sweep
    'sweep_lp': 21e3,                           # Stop frequency of sweep
    'sweep_amplitude': -40,                     # Output Level in dBFS
    # 'sweep_win_in_periods': 1,
    # 'sweep_win_out_periods': 6,
    'sweep_duration': [0.2, 4.0, 0.8],          # Sweep Duration. silence, sweep, silence
    'fs': fs                                    # Sampling Rate
}
sweep = capture.generate_sweep(sweep_config=sweep_config)

deconv_config = {"deconv_type": "lin",          # Decovolution type. "lin" or "cyc"
                 "deconv_hp": 50,               # Lower bandpass Frequency
                 "deconv_lp": 18e3,             # Upper Bandpass Frequency
                 "deconv_inv_dyn": None,
                 "deconv_ir_len_s": 0.6,
                 "deconv_win_in_len": 64,
                 "deconv_win_out_len": 64,
                 "deconv_ir_amp": 0
                 }

sofa_filename = '../Data/HRTF/sofa/HRIR_FULL2DEG.sofa'

hrtf = sofa.Database.open(sofa_filename)  # Load sofa file into hrtf_data

# Load audio in and resample if necessary
audio_filename = '../Data/SoundFiles/Drums2_48.wav'
audio_in = utils.resample(audio_filename=audio_filename,
                          target_fs=fs)

audio_in = audio_in[:fs*4]
mic_locs_filename = '../Data/mic_layouts/micLocs6.mat'  # load mat-file containing SDM microphone coordinates
mic_locs = utils.load_mic(mic_locs_filename)
mic_locs_sph, mic_locs_sph_deg = utils.cart2sph(mic_locs)
numMics = np.size(mic_locs)

print('To record sweep type "rec", to load previous recording type "load"')
rec_load = input()

render_brirs = False  # Render Binaural Room Impulse Response
rotate = False       # Rotate DOAs for Soundscape Renderer
listen = True        # Listen Directly to the auralized signal
smooth_doa = True    # Apply DOA smoothing
align = True         # Align Direct sound to a given angle

doa_filepath = '../Data/DOA/'
brir_filepath = "../Data/BRIRs/"

"""
Load or record Impulse responses. 
"""
if rec_load == 'rec':  # Record sweep and extract impulse responses
    meas_dict, meas_name = capture.record(sweep, fs, device_name, input_idx, output_idx, deconv_config)
    ir_spatial = meas_dict['ir_spatial']
    ir_omni = meas_dict['ir_omni']
    sweep = meas_dict['sweep']
    bgbgb = input('wait')
    ir_filename_to_load = meas_name
    sdm_check = True
elif rec_load == 'load':     # Load a previously recorded ir
    print("Input name of ir to import: ")
    ir_filename_to_load = input()
    loadpath = 'Data/IRs/'
    loadFile = loadpath + ir_filename_to_load
    ir = loadmat(loadFile)
    number_samples = int(0.6*fs)
    ir_omni = ir['ir_omni']
    ir_omni = ir_omni[:, :number_samples]
    ir_spatial = ir['ir_spatial']
    ir_spatial = ir_spatial[:, :number_samples]
    num_mics = ir_spatial.shape[2]
    sweep_sig = ir['sweep']
    num_channels = ir_spatial.shape[0]
    assert(len(ir_omni.shape) == 2)

    ir_spatial = analysis.align_ir(ir_omni=ir_omni, ir_spatial=ir_spatial)
    ir_spatial = np.reshape(ir_spatial, (number_samples, num_mics))
    ir_tot = np.zeros((number_samples, 7))

    ir_tot[:, 0] = ir_omni.flatten()
    ir_dict = {'ir_omni': ir_omni,
               'ir_spatial': ir_spatial}
    # Assign ir_spatial to the remaining columns of ir_tot

    cut = False
    sdm_check = True
    if cut:
        ir_spatial = utils.find_max_and_set_zero(ir_spatial, num_indices=10)
        ir_omni = utils.find_max_and_set_zero(ir_omni, num_indices=10)
    ir_dict = {'ir_omni': ir_omni,
               'ir_spatial': ir_spatial,
               'fs': fs}

else:
    print("Input name of DOAs to import: ")
    doa_filename = input()
    loadpath = 'Data/Simulation_Data/'

    print("Input name of IR to import: ")
    ir_filename_to_load = input()
    loadFile = doa_filename
    doa = loadmat(loadFile)
    ir = loadmat(ir_filename_to_load)
    DOA = doa['DOA']
    ir_omni = ir['irCenter']
    sdm_check = False
    bbb = input("wait")
    fs = 48000
    smooth_order = 1


brir_len = ir_omni.shape[1] + hrtf.Dimensions.N
len_ir = ir_omni.shape[1]
DOA_rot_rad = np.zeros([len_ir, 3])


angle_lim = 90
window_length = 32


if sdm_check:
    ir_doa = ir_spatial
    ir_doa = analysis.bp_filt(data=ir_doa, fs=fs, highcut=5000, lowcut=300)
    print(f"\nBuilding SDM Struct ... ", end="")
    a = analysis.create_sdm_dict(mic_locs,
                                 show_array=False,
                                 win_len=window_length
                                 )
    print("SDM struct built.")
    # plot.plot_time_signals(ir_doa)
    print(f"\nEstimating Directions of arrival ... ", end="")
    DOA = analysis.sdm_par(ir_doa, a)
    DOA_rot_rad, DOA_rot_deg = utils.cart2sph(DOA)

    if align:
        DOA, aligned = analysis.align_DOA2(IR=ir_omni, DOAs_cartesian=DOA, angle=0, tolerance=1)
        DOA_rot_rad, DOA_rot_deg = utils.cart2sph(DOA)

    if smooth_doa:
        smooth_order = 6
        DOA[:, 0] = analysis.running_mean(DOA[:, 0], N=smooth_order)
        DOA[:, 1] = analysis.running_mean(DOA[:, 1], N=smooth_order)
        DOA[:, 2] = analysis.running_mean(DOA[:, 2], N=smooth_order)
        sma = '_sma'
    else:
        smooth_order = 1
        sma = ''

    DOA_rot_rad, DOA_rot_deg = utils.cart2sph(DOA)
    print("Directions of arrival Estimated.")

    if align:
        align = '_align'
    else:
        align = ''

    doa_dict = {'DOA': DOA,
                'fs': fs,
                'p': ir_omni,
                'window_length': window_length,
                'mic_locs': mic_locs,
                'numMics': numMics
                }
    savemat(f'Data/DOA/DOA_{ir_filename_to_load}.mat', doa_dict)

if render_brirs:
    if rotate:
        angles = np.arange(-angle_lim, angle_lim + 1, 1)
        len_angles = angles.shape[0]
        # angles = np.flip(angles)
    else:
        angles = np.array([0])
        len_angles = 0
    brirs = np.zeros((360, 2, brir_len))
    for j, angle in enumerate(angles):
        # DOA_sph_rad[:, 0] +=

        DOA_rot = utils.rotate_cartesian(DOA, -angle)
        print(angle)
        if j % 1000 == 0:
            DOA_rot_rad, DOA_rot_deg = utils.cart2sph(DOA_rot)
            plot.doa(azi=DOA_rot_rad[:, 0],
                     colat=DOA_rot_rad[:, 1],
                     fs=fs,
                     p=ir_omni,
                     win_len=window_length,
                     title=f"DOAs at w = {window_length} and N = {smooth_order}, meas: {ir_filename_to_load}")

        print(f"\nGenerate Binaural Room Impulse Response for \n Head Orientation at " + str(angle) + "... ", end="")
        brir_L, brir_R = analysis.render_brir(hrtf, DOA_rot, ir_omni[0, :])

        brirs[angle, 0, :-1] = brir_L
        brirs[angle, 1, :-1] = brir_R

    max_value = np.max(np.abs(brirs))
    # Normalize the impulse responses in the third dimension
    brirs_norm = brirs / max_value

    brirs_norm *= 10 ** (-10 / 20)  # as factor
    print("Done")
    if rec_load == 'rec':
        ir_filename_to_load = meas_name
    if smooth_doa:
        brir_filename = f"{ir_filename_to_load}_brir_{window_length}_sma{align}.wav"
    else:
        brir_filename = f"{ir_filename_to_load}_brir_{window_length}{align}.wav"

    if rotate:
        utils.write_SSR_IRs(brir_filepath+brir_filename,
                            brirs_norm[:, 0, :],               # Left Ear
                            brirs_norm[:, 1, :],               # Right Ear
                            wavformat="float32",
                            fs=fs)

    print(f"\nSaving Binaural Room Impulse Response... ", end="")
    brir_dict = {'BRIR': brirs,
                 'fs': fs,
                 'DOA': DOA,
                 'window_length': window_length,
                 'mic_locs': mic_locs,
                 'numMics': numMics
                 }
    brir_path = '../Data/BRIRs/'
    brir_filename = 'brirTest.mat'
    savemat(brir_path + brir_filename, brir_dict)
    print("Done")

else:
    print(f"\nGenerating Loudspeaker Signals and saving to wav... ", end="")
    loudspeaker_filepath = '../Data/Loudspeaker_signals/'
    listener_position = [0, 0, 0]
    ls_setup = spa.io.load_layout("../Data/ls_layouts/Atmos.json", listener_position=listener_position)
    ls_setup.show()
    spa.plot.hull(ls_setup)
    spa.plot.hull_normals(ls_setup)
    rendered_ls_sig = np.zeros([len(ls_setup.ls_gains),len(audio_in)])
    gains_nls = spa.decoder.nearest_loudspeaker(DOA, ls_setup)

    ls_ir = ls_setup.loudspeaker_signals(ls_gains=gains_nls, sig_in=ir_omni)
    ls_ir = utils.normalize(ls_ir)

    utils.write_to_wav(filename=loudspeaker_filepath+'ls_ir_atmos_test_far.wav', data=ls_ir, samplerate=fs)

    for i in range(ls_ir.shape[0]):
        rendered_ls_sig[i, :] = np.convolve(audio_in, ls_ir[i], mode='same')
        # print("max: ", np.max(rendered_ls_sig[i, :]))

    norm_ls_sigs = utils.normalize(rendered_ls_sig)

    rendered_filename = 'rendered_vbap.wav'
    utils.write_to_wav(filename=loudspeaker_filepath+rendered_filename,
                       data=rendered_ls_sig,
                       samplerate=fs)
    print("Done")


if listen:
    if render_brirs:

        left_channel = sig.convolve(audio_in, brir_L, mode='full', method='auto')
        right_channel = sig.convolve(audio_in, brir_R, mode='full', method='auto')

        # Normalize the output signal by applying a gain
        left_channel = utils.normalize(left_channel)
        right_channel = utils.normalize(right_channel)
        binaural = np.column_stack((left_channel, right_channel))

        wavfile.write(f'Data/BinSignals/Drums_win{window_length}_smooth{smooth_order}.wav', fs, binaural)

        dev_name = 'Externa hörlurar'
        print("Playing binaural signal...")
        sd.play(binaural, fs, device=dev_name)
        sd.wait()
        print("Done")
    else:
        ir_omni = ir_omni.T
        audio_conv = sig.convolve(audio_in, ir_omni[:,0])

        audio_conv *= 10 ** (-20 / 20)  # as factor

        dev_name = 'Externa hörlurar'
        print("Playing audio signal...")
        sd.play(audio_in, fs, device=dev_name)
        sd.wait()
        print("Done")

        print("Playing binaural signal...")
        sd.play(audio_conv, fs, device=dev_name)
        sd.wait()
        print("Done")

print('Program executed in', np.floor((time.time() - start_time)), 'seconds.')
