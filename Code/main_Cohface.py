import sys
import cv2
from numpy.distutils.pathccompiler import PathScaleCCompiler
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from npy import motion_signal_extract, apply_filter,normalize_data,calculate_correlation, RespirationRateCalculator,Noise_Removal_Module, calculate_vertical_distance
from sklearn.decomposition import PCA,FastICA
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, freqs
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial
import h5py
from scipy.stats import pearsonr


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def compute_combined_main_frequency(freqs, magnitudes, primary_ratio_threshold=0.5, top_n=3, energy_threshold=0.707):
    total_energy = np.sum(magnitudes)
    max_magnitude = np.max(magnitudes)
    primary_ratio = max_magnitude / total_energy
    if primary_ratio <= primary_ratio_threshold:
        threshold = max_magnitude * energy_threshold
        indices_above_threshold = np.where(magnitudes >= threshold)[0]
        selected_freqs = freqs[indices_above_threshold]
        selected_magnitudes = magnitudes[indices_above_threshold]
        method_used = "threshold"
    else:
        top_indices = np.argsort(magnitudes)[-top_n:]
        selected_freqs = freqs[top_indices]
        selected_magnitudes = magnitudes[top_indices]
        method_used = "top_n"
    combined_frequency = np.sum(selected_freqs * selected_magnitudes) / np.sum(selected_magnitudes)
    return combined_frequency, method_used,primary_ratio

def classify_signal(signal, threshold=0.6):
    peaks, _ = find_peaks(signal)
    valleys, _ = find_peaks(-signal)
    if len(peaks) > 2:
        peak_intervals = np.diff(peaks)
        delta_n = np.diff(peak_intervals)
        signal_length = peaks[-1] - peaks[0]  # 信号长度
        D = (np.mean(np.abs(delta_n)) / (signal_length / len(peaks))) * 10
        K = np.log(1 + len(peaks)) * (signal_length / 100)
        periodicity_score = (3) / (3 + D)
    else:
        periodicity_score = 0

    peak_amplitudes = []
    for i, peak in enumerate(peaks):
        left_valley = signal[peak - 1] if peak > 0 else signal[peak]
        right_valley = signal[peak + 1] if peak < len(signal) - 1 else signal[peak]
        if peak == 0:
            peak_amplitude = abs(signal[peak] - right_valley)
        elif peak == len(signal) - 1:
            peak_amplitude = abs(signal[peak] - left_valley)
        else:
            peak_amplitude = min(abs(signal[peak] - left_valley), abs(signal[peak] - right_valley))
        peak_amplitudes.append(peak_amplitude)
    sorted_amplitudes = sorted(peak_amplitudes)
    if len(sorted_amplitudes) > 2:
        trimmed_amplitudes = sorted_amplitudes[1:-1]
    else:
        trimmed_amplitudes = sorted_amplitudes
    mean_amplitude = np.mean(trimmed_amplitudes)
    std_amplitude = np.std(trimmed_amplitudes)
    z_scores = [(amp - mean_amplitude) / std_amplitude if std_amplitude > 0 else 0 for amp in peak_amplitudes]
    max_z_score = np.max(np.abs(z_scores))
    abnormal_amplitude_score = 3 / (3 + max_z_score)

    scores = [periodicity_score, abnormal_amplitude_score]
    total_score = np.mean(scores)
    answer = (total_score >= threshold)
    return answer, total_score, scores

def spectral_subtraction_denoise_single(y_signal, x_signal, fs=20):
    min_length = min(len(y_signal), len(x_signal))
    y_signal = y_signal[:min_length]
    x_signal = x_signal[:min_length]
    Y_fft = fft(y_signal)
    X_fft = fft(x_signal)
    freqs = np.fft.fftfreq(len(x_signal), d=1 / fs)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = np.abs(X_fft)[:len(freqs) // 2]
    combined_main_frequency, method_used,primary_ratio = compute_combined_main_frequency(
        positive_freqs, positive_magnitude, primary_ratio_threshold=0.7, top_n=3, energy_threshold=0.5
    )
    beta = np.abs(freqs) / combined_main_frequency
    beta[np.isnan(beta)] = 0
    noise_amplitude = np.abs(X_fft)
    signal_amplitude = np.abs(Y_fft)
    clean_amplitude = signal_amplitude - beta*noise_amplitude
    clean_amplitude[clean_amplitude < 0] = 0
    clean_fft = clean_amplitude * np.exp(1j * np.angle(Y_fft))
    denoised_signal = np.real(ifft(clean_fft))

    return denoised_signal

def zero_padding_extend(signal_group, padding_factor=2):
    if signal_group.ndim == 1:

        num_frames = len(signal_group)
        padded_length = num_frames * padding_factor
        signal_padded = np.pad(signal_group, (0, padded_length - num_frames), mode='constant')
        return signal_padded
    elif signal_group.ndim == 2:
        num_frames, num_signals = signal_group.shape
        padded_length = num_frames * padding_factor
        extended_signals = np.zeros((padded_length, num_signals))
        for col in range(num_signals):
            signal = signal_group[:, col]
            signal_padded = np.pad(signal, (0, padded_length - num_frames), mode='constant')
            extended_signals[:, col] = signal_padded
    return extended_signals

def zero_padding_restore(signal_group, original_length):
    if signal_group.ndim == 1:
        restored_signal = signal_group[:original_length]
        return restored_signal
    elif signal_group.ndim == 2:
        padded_length, num_signals = signal_group.shape
        restored_signals = np.zeros((original_length, num_signals))
        for col in range(num_signals):
            restored_signals[:, col] = signal_group[:original_length, col]
        return restored_signals
    else:
        raise ValueError("should be 1D or 2D。")

def ica_time_frequency_separation(signal_group_1, signal_group_2, noise_reference):
    num_frames, num_signals = signal_group_1.shape
    separated_signals_group = np.zeros_like(signal_group_1)
    for col in range(num_signals):
        signal_1 = signal_group_1[:, col]
        signal_2 = signal_group_2[:, col]
        mixed_signals_time = np.vstack((signal_1, signal_2)).T
        ica_time = FastICA(n_components=2, max_iter=1000, random_state=0)
        separated_signals_time = ica_time.fit_transform(mixed_signals_time)
        fft_signal_1 = fft(signal_1)
        fft_signal_2 = fft(signal_2)
        mixed_signals_freq = np.vstack((np.abs(fft_signal_1), np.abs(fft_signal_2))).T
        ica_freq = FastICA(n_components=2, max_iter=1000, random_state=0)
        separated_signals_freq = ica_freq.fit_transform(mixed_signals_freq)
        fft_noise_reference = np.abs(fft(noise_reference))
        corr_time_1 = np.corrcoef(separated_signals_time[:, 0], noise_reference)[0, 1]
        corr_time_2 = np.corrcoef(separated_signals_time[:, 1], noise_reference)[0, 1]
        time_score_1 = abs(corr_time_1)
        time_score_2 = abs(corr_time_2)
        if time_score_1 < time_score_2:
            best_time_signal = separated_signals_time[:, 0]
        else:
            best_time_signal = separated_signals_time[:, 1]
        corr_freq_1 = np.corrcoef(np.abs(separated_signals_freq[:, 0]), fft_noise_reference)[0, 1]
        corr_freq_2 = np.corrcoef(np.abs(separated_signals_freq[:, 1]), fft_noise_reference)[0, 1]
        freqs_score_1 = abs(corr_freq_1)
        freqs_score_2 = abs(corr_freq_2)
        if freqs_score_1 < freqs_score_2:
            best_freq_signal = separated_signals_freq[:, 0]
            worst_freq_signal = separated_signals_freq[:, 1]
        else:
            best_freq_signal = separated_signals_freq[:, 1]
            worst_freq_signal = separated_signals_freq[:, 0]
        best_freq_signal_time = np.real(ifft(best_freq_signal + 1j * np.angle(fft_signal_1)))
        worst_freq_signal_time = np.real(ifft(worst_freq_signal + 1j * np.angle(fft_signal_1)))
        fused_signal = 0.5*best_freq_signal_time + 0.5*best_time_signal
        separated_signals_group[:, col] = fused_signal
    return separated_signals_group

def process_signal(y_signal, x_signal, y_source,x_source, sampling_rate):
    N = len(y_signal)
    freq = np.fft.fftfreq(N, d=1 / sampling_rate)
    y_spectrum = np.fft.fft(y_signal)
    y_amplitude = np.abs(y_spectrum)
    y_main_freq_idx = np.argmax(y_amplitude[:N // 2])
    y_main_freq = freq[y_main_freq_idx]
    y_main_amp = y_amplitude[y_main_freq_idx]
    y_threshold_3db = y_main_amp / np.sqrt(2)
    significant_freqs = freq[:N // 2][y_amplitude[:N // 2] >= y_threshold_3db]
    not_has_other_significant_freqs = not any(
        (y_amplitude[i] >= y_threshold_3db) and (i != y_main_freq_idx)
        for i in range(len(y_amplitude[:N // 2]))
    )and y_main_freq>0.167
    if not_has_other_significant_freqs and not any(f > 0.55 for f in significant_freqs):
        reconstructed_spectrum = np.zeros_like(y_spectrum)
        reconstructed_spectrum[y_main_freq_idx] = y_spectrum[y_main_freq_idx]
        reconstructed_spectrum[-y_main_freq_idx] = y_spectrum[-y_main_freq_idx]
        reconstructed_signal = np.fft.ifft(reconstructed_spectrum).real
        RR = y_main_freq * 60
        method = 0
    elif any(f > 0.55 for f in significant_freqs):
        method = '0'
        RR = ''
        noise_reference = highpass_filter(x_source, cutoff_bpm=10)
        y_denoise = ica_time_frequency_separation(noise_reference,y_source,x_signal)
        reconstructed_signal = pca_signal_fusion(y_denoise)
        reconstructed_signal = highpass_filter(reconstructed_signal, cutoff_bpm=10)
    elif not not_has_other_significant_freqs and not any(f > 0.55 for f in significant_freqs):
        reconstructed_signal = spectral_subtraction_denoise_single(y_signal, x_signal, fs=20)
        method = '1'
        RR = ''
    return reconstructed_signal, method, RR

def extract_main_frequencies(signal, fs, num_frequencies=3, threshold_ratio=0.1):

    N = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    magnitude = np.abs(fft_result)
    positive_freqs = freqs[:N // 2]
    positive_magnitude = magnitude[:N // 2]
    threshold = np.max(positive_magnitude) * threshold_ratio
    significant_indices = np.where(positive_magnitude > threshold)[0]
    significant_frequencies = positive_freqs[significant_indices]
    significant_magnitudes = positive_magnitude[significant_indices]
    freq_magnitude_pairs = list(zip(significant_frequencies, significant_magnitudes))
    sorted_pairs = sorted(freq_magnitude_pairs, key=lambda x: x[1], reverse=True)[:num_frequencies]
    sorted_frequencies = [pair[0] for pair in
                          sorted(freq_magnitude_pairs, key=lambda x: x[1], reverse=True)[:num_frequencies]]
    return sorted_frequencies, sorted_pairs


def highpass_filter(signal, cutoff_bpm, fs=20, order=2):
    cutoff_hz = cutoff_bpm / 60.0
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    if isinstance(signal, np.ndarray) and signal.ndim == 2:
        filtered_signal = np.apply_along_axis(lambda col: filtfilt(b, a, col), axis=0, arr=signal)
    elif isinstance(signal, np.ndarray) and signal.ndim == 1:
        filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def lowpass_filter(signal, cutoff_bpm, fs=20, order=2):
    cutoff_hz = cutoff_bpm / 60.0
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if isinstance(signal, np.ndarray) and signal.ndim == 2:
        filtered_signal = np.apply_along_axis(lambda col: filtfilt(b, a, col), axis=0, arr=signal)
    elif isinstance(signal, np.ndarray) and signal.ndim == 1:
        filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = []
    frame_indices = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        selected_frames.append(first_frame)
        frame_indices.append(0)
    three_seconds_frame_index = int(fps * 3)
    if three_seconds_frame_index < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, three_seconds_frame_index)
        ret, three_seconds_frame = cap.read()
        if ret:
            selected_frames.append(three_seconds_frame)
            frame_indices.append(three_seconds_frame_index)
    cap.release()
    temp_video_path = os.path.join(output_dir, "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = selected_frames[0].shape[:2]
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    for frame in selected_frames:
        out.write(frame)
    out.release()
    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(temp_video_path, det_dim=416, num_peroson=1, gen_output=True)
    os.remove(temp_video_path)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = keypoints[0]
    keypoints_with_info = []
    if len(keypoints) > 0:
        hrnet_frame_index = 0
        frame_keypoints = keypoints[hrnet_frame_index]
        frame = selected_frames[hrnet_frame_index]
        left_shoulder_coords = frame_keypoints[11][:2]
        right_shoulder_coords = frame_keypoints[14][:2]
        cv2.circle(frame, (int(left_shoulder_coords[0]), int(left_shoulder_coords[1])), 2, (0, 165, 255), -1)  # 左肩橙色
        cv2.circle(frame, (int(right_shoulder_coords[0]), int(right_shoulder_coords[1])), 2, (0, 165, 255), -1)  # 右肩橙色
        midpoint_x = (left_shoulder_coords[0] + right_shoulder_coords[0]) / 2
        midpoint_y = (left_shoulder_coords[1] + right_shoulder_coords[1]) / 2
        central_point = (midpoint_x, midpoint_y)
        distance = np.linalg.norm(left_shoulder_coords - right_shoulder_coords)
        keypoints_with_info.append({
            "frame_index": frame_indices[hrnet_frame_index],
            "left_shoulder": left_shoulder_coords,
            "right_shoulder": right_shoulder_coords,
            "central_point": central_point,
            "distance": distance
        })
    return keypoints_with_info

def gradient_based_canny_threshold(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_std = np.std(gradient_magnitude)
    lower = int(max(0, grad_std * 0.5))
    upper = int(min(255, grad_std * 1.5))
    return lower, upper

def Edge_Enhancement_RGB(Image, center_point, side_length):
    center_x = int(round(center_point[0]))
    center_y = int(round(center_point[1]))
    half_length = int(round(side_length / 3))
    half_width = int(round(side_length / 5))
    x_start = max(center_x - half_length, 0)
    y_start = max(center_y - half_width-15, 0)
    x_end = min(center_x + half_length, Image.shape[1])
    y_end = min(center_y + half_width, Image.shape[0])
    Image_sub = Image[y_start:y_end, x_start:x_end]
    old_gray = cv2.cvtColor(Image_sub, cv2.COLOR_BGR2GRAY)
    if len(Image_sub.shape) == 3:
        R_channel = Image_sub[:, :, 0]
        G_channel = Image_sub[:, :, 1]
        B_channel = Image_sub[:, :, 2]
    else:
        raise ValueError("should be RGB Image")
    # 确保每个通道的数据类型是 uint8
    R_channel = np.uint8(R_channel)
    G_channel = np.uint8(G_channel)
    B_channel = np.uint8(B_channel)
    low_threshold_R, high_threshold_R = gradient_based_canny_threshold(R_channel)
    low_threshold_G, high_threshold_G = gradient_based_canny_threshold(G_channel)
    low_threshold_B, high_threshold_B = gradient_based_canny_threshold(B_channel)
    R_edges = cv2.Canny(R_channel, low_threshold_R, high_threshold_R)
    G_edges = cv2.Canny(G_channel, low_threshold_G, high_threshold_G)
    B_edges = cv2.Canny(B_channel, low_threshold_B, high_threshold_B)
    Combined_edges = cv2.bitwise_and(cv2.bitwise_and(R_edges, G_edges), B_edges)
    Combined_edges[0, :] = 0
    Combined_edges[-1, :] = 0
    Combined_edges[:, 0] = 0
    Combined_edges[:, -1] = 0
    return Combined_edges, Image_sub, x_start, y_start

def FeaturePointSelectionInRegion(Image, center_point, side_length, FPN=25):
    Image_edges, Image_gray, x_start, y_start = Edge_Enhancement_RGB(Image, center_point, side_length
                                                                     )
    feature_params = dict(maxCorners=0,
                          qualityLevel=0.2,
                          minDistance=5)
    p0 = cv2.goodFeaturesToTrack(Image_edges, mask=None, **feature_params, useHarrisDetector=False, blockSize=3)
    if p0 is None:
        return np.empty((0, 1, 2), dtype=np.float32), center_point
    corners = np.squeeze(p0)  # 去除冗余维度
    corners_with_response = []
    for i, corner in enumerate(corners):
        x, y = corner
        response = p0[i, 0, 0]
        corners_with_response.append((response, x, y))
    sorted_corners = sorted(corners_with_response, key=lambda c: c[0], reverse=True)
    selected_corners = sorted_corners[:FPN]
    actual_FPN = min(len(sorted_corners), FPN)
    FPMap = np.zeros((actual_FPN, 1, 2), dtype=np.float32)
    for i, (_, x, y) in enumerate(selected_corners):
        FPMap[i, 0, 0] = x + x_start
        FPMap[i, 0, 1] = y + y_start
    return FPMap, center_point

def process_feature_mtx(feature_mtx):
    processed_mtx = np.zeros_like(feature_mtx)
    for i in range(feature_mtx.shape[1]):
        filtered_signal = apply_filter(feature_mtx[:, i], filter_order=3, low_pass=10/60, high_pass=50/60,fs=20)
        processed_mtx[:, i] = normalize_data(filtered_signal)
    return processed_mtx

def pca_signal_fusion(signals, n_components=1):
    signals = np.array(signals)
    if signals.ndim != 2:
        raise ValueError("Input signals must be a 2D array.")
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(signals)
    fused_signal = principal_components[:, 0]
    return fused_signal

def ImproveOpticalFlow(video_path, Kpt_Informationh, output_dir, RR_Evaluation=True):
    cap = cv2.VideoCapture(video_path)
    fs = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    RespCurves = []
    FrameSets = []
    MotionCurves = []
    Source_signal = []
    method = []
    Q_Score_write = []
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

    for idx, info in enumerate(Kpt_Informationh):
        start_frame = info["frame_index"]
        center_point = np.array(info["central_point"], dtype=np.float32)
        distance = info["distance"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, old_frame = cap.read()
        if not ret:
            raise ValueError(f"Error reading frame {start_frame} from video.")
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        P2, center_center = FeaturePointSelectionInRegion(
            Image=old_frame,
            FPN=50,
            side_length=distance,
            center_point=center_point,
            Output_path=output_dir,
            Frame_ID=start_frame
        )

        FeatureMtx_C = np.zeros((total_frame - start_frame, P2.shape[0], 2))
        FeatureMtx_C[0, :, :] = P2[:, 0, :]
        color = colors[idx % len(colors)]
        lk_params = dict(winSize=(16, 16), maxLevel=10)
        frame_num = 1

        while start_frame + frame_num < total_frame:
            frame_num += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            P2_new, st2, err2 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, P2, None, **lk_params)
            old_gray = frame_gray.copy()
            P2 = P2_new.reshape(-1, 1, 2)
            FeatureMtx_C[frame_num - 1, :, :] = P2[:, 0, :]
            FeatureMtx_Y_C = FeatureMtx_C[:, :, 1]
            FeatureMtx_X_C = FeatureMtx_C[:, :, 0]

        FeatureMtx_Y_C = process_feature_mtx(FeatureMtx_Y_C)
        RespCurve_Y_C = pca_signal_fusion(FeatureMtx_Y_C)
        FeatureMtx_X_C = process_feature_mtx(FeatureMtx_X_C)
        RespCurve_X_C = pca_signal_fusion(FeatureMtx_X_C)
        RespCurve_X_C_highpass = highpass_filter(RespCurve_X_C, cutoff_bpm=30)

        Q_Score,total_score,scores = classify_signal(RespCurve_Y_C)
        Denoise_output, method_use,RR =process_signal(RespCurve_Y_C,RespCurve_X_C_highpass,FeatureMtx_Y_C,FeatureMtx_X_C,sampling_rate=30)
        Denoise_output=apply_filter(Denoise_output,fs=20)
        RespCurves.append(Denoise_output)
        method.append(method_use)
        Q_Score_write.append(Q_Score)
        Source_signal.append(RespCurve_Y_C)# 保留实际呼吸信号
        FrameSets.append(np.arange(start_frame, start_frame + len(RespCurve_Y_C)))
        MotionCurves.append(RespCurve_X_C_highpass)
    cap.release()
    if RR_Evaluation:
        return RespCurves, MotionCurves, Source_signal, method,Q_Score_write ,FrameSets,scores,RR
    else:
        return RespCurves, MotionCurves, Source_signal, method,Q_Score_write ,FrameSets,scores,RR


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def z_score_outliers(values, threshold=3):
    mean = np.mean(values)
    std = np.std(values)
    z_scores = np.abs((values - mean) / std)
    outliers = np.where(z_scores > threshold)[0]
    return outliers


def amplify_signal(signal1, signal2, output_path, min_peak_distance=15, min_gain_threshold=0.01, gain_base=1,
                   poly_order=3):
    min_length = min(len(signal1), len(signal2))
    signal1 = signal1[:min_length]
    signal2 = signal2[:min_length]
    peaks2, _ = find_peaks(signal2, distance=min_peak_distance)
    troughs2, _ = find_peaks(-signal2, distance=min_peak_distance)
    critical_points = np.sort(np.concatenate((peaks2, troughs2)))
    differences = np.abs(np.diff(signal2[critical_points]))
    outliers = z_score_outliers(differences, threshold=2.5)
    if len(outliers) == 0:
        return signal2

    # 波峰拟合
    peaks, _ = find_peaks(signal1, distance=min_peak_distance)
    peak_x = peaks
    peak_y = signal1[peaks]
    valid_peaks = peak_y > 0
    peak_x = peak_x[valid_peaks]
    peak_y = peak_y[valid_peaks]
    if len(peak_x) < poly_order + 1:
        raise ValueError("Insufficient valid peaks in signal1 for fitting.")
    poly_coeffs = Polynomial.fit(peak_x, peak_y, poly_order)
    fitted_curve = poly_coeffs(np.arange(len(signal1)))
    fitted_curve = np.maximum(fitted_curve, min_gain_threshold)
    final_value = fitted_curve[-1]
    gain_curve = (gain_base * final_value) / fitted_curve
    critical_x = np.concatenate(([0], critical_points, [len(signal2) - 1]))
    critical_y = signal2[critical_x]
    critical_gain = gain_curve[critical_x]
    critical_gain[0] = 1
    critical_gain[-1] = 1
    if len(critical_gain) > 3:
        max_first_peak_gain = critical_gain[4]
        if critical_gain[1] > max_first_peak_gain:
            critical_gain[1] = max_first_peak_gain
        max_first_trough_gain = critical_gain[4]
        if critical_gain[2] > max_first_trough_gain:
            critical_gain[2] = max_first_trough_gain
    amplified_critical_y = critical_y * critical_gain
    interpolation = interp1d(
        critical_x,
        amplified_critical_y,
        kind='quadratic',
        fill_value="extrapolate"
    )
    amplified_signal = interpolation(np.arange(len(signal2)))
    return amplified_signal

def process_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as hdf:
        if 'respiration' in hdf:
            data = np.array(hdf['respiration'])
        else:
            raise KeyError(f"flie {file_path} not include 'respiration'")
    return data

def calculate_fundamental_freq(signal, fs, min_freq=2, max_freq=40):
    freq_spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs) * 60
    valid_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    freq_spectrum = freq_spectrum[valid_indices]
    freqs = freqs[valid_indices]
    fundamental_freq = freqs[np.argmax(np.abs(freq_spectrum))]
    return fundamental_freq

def calculate_snr(signal, fundamental_freq, fs, bandwitch = 10):
    freq_spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs) * 60
    valid_indices = np.where((freqs >= 2) & (freqs <= 40))[0]
    freq_spectrum = freq_spectrum[valid_indices]
    freqs = freqs[valid_indices]
    Ut = np.zeros_like(freqs)
    harmonic_freqs = [fundamental_freq, 2 * fundamental_freq]
    for harmonic in harmonic_freqs:
        harmonic_idx = np.where((freqs >= harmonic - bandwitch) & (freqs <= harmonic + bandwitch))[0]
        Ut[harmonic_idx] = 1
    signal_energy = np.sum((Ut * np.abs(freq_spectrum)) ** 2)
    noise_energy = np.sum(((1 - Ut) * np.abs(freq_spectrum)) ** 2)
    snr = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else -np.inf
    return snr
def process_video(video_path, output_dir, result_df):
    folder_name = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.basename(video_path).split('.')[0]
    output_data_dir = os.path.join(output_dir, f'{folder_name}/{video_name}/data.xlsx')
    ensure_dir(os.path.dirname(output_data_dir))
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    Kpt_information = get_pose2D(video_path, output_dir)
    output_image_dir = os.path.dirname(output_data_dir)
    Center_optical_signal, Motion_signal, Source_signal, method_use, Signal_quality, total_frame, scores, RR = ImproveOpticalFlow(
        video_path, Kpt_information, output_image_dir)
    optical_amplified_signal = amplify_signal(Motion_signal[0], Center_optical_signal[0], output_image_dir)
    _, Y_frequency = extract_main_frequencies(Source_signal[0], fs=20)
    _, X_frequency = extract_main_frequencies(Motion_signal[0], fs=20)

    if len(Source_signal[0]) < len(total_frame[0]):
        total_frame[0] = total_frame[0][:len(Source_signal[0])]

    Center_optical_signal_RR = RespirationRateCalculator(Center_optical_signal[0], 20, len(Center_optical_signal[0]))
    Center_optical_signal_NFCP = Center_optical_signal_RR.NegativeFeedbackCrossoverPointMethod()

    Source_signal_RR = RespirationRateCalculator(Source_signal[0], 20, len(Source_signal[0]))
    Source_signal_NFCP = Source_signal_RR.NegativeFeedbackCrossoverPointMethod()

    method_write = method_use[0]
    if method_write!= 0:
        Center_optical_signal_NFCP = Center_optical_signal_NFCP
    else:
        Center_optical_signal_NFCP = Y_frequency[0][0]*60
    Signal_quality_write = Signal_quality[0]
    hdf5_file_path = os.path.join(os.path.dirname(video_path), f'{video_name}.hdf5')
    if os.path.exists(hdf5_file_path):
        print(f'HDF5 file found: {hdf5_file_path}')
    else:
        print(f'HDF5 file not found for video: {video_path}')
        hdf5_file_path = None
    waveform = process_hdf5_file(hdf5_file_path)
    waveform = apply_filter(waveform, filter_order=3, low_pass=10 / 60, high_pass=50 / 60, fs=256)
    waveform = normalize_data(waveform)
    optical_signal_hdf5 = RespirationRateCalculator(waveform, 256, len(waveform))
    breathing_rate_hdf5 = optical_signal_hdf5.NegativeFeedbackCrossoverPointMethod()
    len_waveform = len(waveform)
    time_center = np.arange(len(Center_optical_signal[0])) / 30
    time_target = np.linspace(time_center[0], time_center[-1], len_waveform)
    interpolator = interp1d(time_center, Center_optical_signal[0], kind='linear', fill_value="extrapolate")
    Center_optical_signal_resampled = interpolator(time_target)
    MAE = abs(breathing_rate_hdf5 - Center_optical_signal_NFCP)
    PCC, _ =pearsonr(waveform, Center_optical_signal_resampled)
    PCC = abs(PCC)
    fundamental_freq = calculate_fundamental_freq(Center_optical_signal[0],fs=30)
    SNR = calculate_snr(Center_optical_signal[0],fundamental_freq,fs=30)
    SNR = abs(SNR)
    MAPE = abs(abs(breathing_rate_hdf5 - Center_optical_signal_NFCP) / breathing_rate_hdf5)
    print(f"MAE:{MAE},    PCC:{PCC},    SNR:{SNR}")
    new_row = pd.DataFrame({
        'folder_name': [folder_name],
        'video_name': [video_name],
        'Frame': [total_frames],
        'PCC':[PCC],
        'MAE':[MAE],
        'SNR':[SNR],
        'MAPE':[MAPE]
    })
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    return result_df
if __name__ == "__main__":
    input_folder = './COHFACE'
    output_folder = './output'
    excel_output_path = os.path.join(output_folder, 'video_results.xlsx')
    result_df = pd.DataFrame(columns=['folder_name', 'video_name', 'Frame'])
    video_files = [os.path.join(root, file)
                   for root, dirs, files in os.walk(input_folder)
                   for file in files if file.endswith('.avi') ]
    for video_path in tqdm(video_files, desc="Processing Videos"):
        result_df = process_video(video_path, output_folder, result_df)
    # 保存结果到Excel文件
    result_df.to_excel(excel_output_path, index=False)
    print(f'Save to {excel_output_path}')