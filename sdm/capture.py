from . import analysis, plot, utils
import sounddevice as sd
import sys
import numpy as np
from scipy.io import savemat
import scipy.signal as sig


def record(sweep, fs, device_name, input_idx, output_idx, deconv_config, num_sources=1):
    """
    Function to record sweep, deconvolve and return a measurement 
    dictionary containing, IRs and measured signal
    :param sweep: Sine sweep signal
    :param fs: 
    :param device_name: 
    :param input_idx: 
    :param output_idx: 
    :param deconv_config:
    :return: 
    """
    deconv_type = deconv_config["deconv_type"]
    deconv_hp = deconv_config["deconv_hp"]
    deconv_lp = deconv_config["deconv_lp"]
    deconv_inv_dyn = deconv_config["deconv_inv_dyn"]
    deconv_ir_len_s = deconv_config["deconv_ir_len_s"]
    deconv_ir_len_sp = deconv_ir_len_s * fs
    deconv_win_in_len = deconv_config["deconv_win_in_len"]
    deconv_win_out_len = deconv_config["deconv_win_out_len"]
    deconv_ir_amp = deconv_config["deconv_ir_amp"]

    ir_spatial = np.zeros((num_sources, int(deconv_ir_len_sp), 6))
    ir_omni = np.zeros((num_sources, int(deconv_ir_len_sp)))

    # fmt: on
    print(f'\nCheck audio device "{device_name}" ... ', end="")
    try:
        sd.check_output_settings(
            device=device_name,
            channels=len(output_idx),  # as number
            samplerate=fs,  # in Hz
        )
        sd.check_input_settings(
            device=device_name,
            channels=len(input_idx),  # as number
            samplerate=fs,  # in Hz
        )
    except (ValueError, sd.PortAudioError) as e:
        sys.exit(e)
    print("done.")

    # IR FOR EACH SPEAKER
    for i in range(num_sources):

        if i == 0:
            output_idx = [1, 3]
        elif i == 1:
            output_idx = [2, 3]
        else:
            break

        while True:
            # Play record
            print(f"\nRecord sweep ... ", end="")
            data_rec = sd.playrec(
                data=sweep,  # as [samples,]
                samplerate=fs,  # in Hz
                device=device_name,
                input_mapping=input_idx,  # channel numbers start from 1
                output_mapping=output_idx,  # channel numbers start from 1
                blocking=False,
            )  # as [channels, samples]
            sd.wait()
            print("done.")

            # check for clipping
            if np.any(np.abs(data_rec) >= 1.0):
                print("Clipping detected, recording again...")
                continue
            break

        # Isolate the loopback signal
        sweep_sig = data_rec[:, 8]

        sweep_sig = sweep_sig.T
        data_rec = data_rec.T

        # Generate Impulse response
        print(f"\nGenerating IR... ", end="")
        ir = analysis.deconvolve(
            exc_sig=sweep_sig,
            rec_sig=data_rec,
            fs=fs,
            f_low=deconv_hp,  # in Hz
            f_high=deconv_lp,  # in Hz
            deconv_type=deconv_type,
            res_len=deconv_ir_len_s,  # in s
            inv_dyn_db=deconv_inv_dyn,  # in dB
            win_in_len=deconv_win_in_len,  # in samples
            win_out_len=deconv_win_out_len,  # in samples
        ).T
        print("Done")

        ir *= 10 ** (deconv_ir_amp / 20)
        print(ir.shape)
        # Separate the center microphone and the satellites
        ir_spatial[i, :, :] = ir[:, :6]
        irr = ir[:1000, 0:6]
        swep = data_rec[:, :6]
        ir_omni[i, :] = ir[:, 7]
        ir_o = ir_omni[:, :500]

        utils.estimate_distance(ir_omni, fs)
        ir_spatial = analysis.align_ir(ir_omni, ir_spatial)
        plot.plot_time_signals(ir_spatial[1,:,:])
    meas_dict = {'ir_spatial': ir_spatial,
                 'ir_omni': ir_omni,
                 'sweep': data_rec,
                 'num_sources': num_sources,
                 'fs': fs,
                 }
    # Save Impulse Response
    print(f"\nSaving Impulse Response ... ", end="")
    meas_name = input("Input IR savename: ")
    savemat('Data/IRs/' + meas_name + '.mat', meas_dict)
    print("done")

    return meas_dict, meas_name


def generate_sweep(sweep_config):
    # Extracting values into variables
    sweep_hp = sweep_config['sweep_hp']
    sweep_lp = sweep_config['sweep_lp']
    sweep_amplitude = sweep_config['sweep_amplitude']
    sweep_duration = sweep_config['sweep_duration']
    fs = sweep_config['fs']

    sweep = sig.chirp(
        t=np.arange(np.ceil(sweep_duration[1] * fs)) / fs,  # in s
        f0=sweep_hp,  # in Hz
        t1=sweep_duration[1],  # in s
        f1=sweep_lp,  # in Hz
        method="logarithmic",  # Sweep Shape
        phi=90,  # in deg (to start with a sample at 0)
    )

    sweep *= 10 ** (sweep_amplitude / 20)  # as factor

    sweep = np.pad(
        array=sweep,
        pad_width=(int(sweep_duration[0] * fs), int(sweep_duration[2] * fs)),
    )

    return sweep
