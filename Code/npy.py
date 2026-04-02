import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
import os

class RespirationRateCalculator:
    def __init__(self, data, fs, N):
        self.data = data
        self.fs = fs
        self.N = N
        self.Time = self.N / fs

    def FFT(self):
        fft_y = fft(self.data)
        maxFrequency = self.fs
        f = np.linspace(0, maxFrequency, self.N)
        abs_y = np.abs(fft_y)
        normalization_y = abs_y / self.N
        normalization_half_y = normalization_y[range(int(self.N / 2))]
        sorted_indices = np.argsort(normalization_half_y)
        RR = f[sorted_indices[-2]] * 60
        return RR

    # # Peak Counting Method
    def PeakCounting(self, Height=None, Threshold=None, MaxRR=45):
        Distance = 60 / MaxRR * self.fs
        peaks, _ = find_peaks(self.data, height=Height, threshold=Threshold, distance=Distance)
        RR = len(peaks) / self.Time * 60
        return RR

    # # Crossover Point Method
    def CrossingPoint(self):
        shfit_distance = int(self.fs / 2)
        data_shift = np.zeros(self.data.shape) - 1
        data_shift[shfit_distance:] = self.data[:-shfit_distance]
        cross_curve = self.data - data_shift

        zero_number = 0
        zero_index = []
        for i in range(len(cross_curve) - 1) :
            if cross_curve[i] == 0 :
                zero_number += 1
                zero_index.append(i)
            else :
                if cross_curve[i] * cross_curve[i + 1] < 0 :
                    zero_number += 1
                    zero_index.append(i)

        cw = zero_number
        N = self.N
        fs = self.fs
        RR1 = ((cw / 2) / (N / fs)) * 60

        return RR1

    def NegativeFeedbackCrossoverPointMethod(self, QualityLevel=0.5):
        shfit_distance = int(self.fs / 2)
        data_shift = np.zeros_like(self.data) - 1
        data_shift[shfit_distance:] = self.data[:-shfit_distance]
        cross_curve = self.data - data_shift

        zero_number = 0
        zero_index = []
        for i in range(len(cross_curve) - 1):
            if cross_curve[i] == 0:
                zero_number += 1
                zero_index.append(i)
            else:
                if cross_curve[i] * cross_curve[i + 1] < 0:
                    zero_number += 1
                    zero_index.append(i)

        cw = zero_number
        RR1 = ((cw / 2) / (self.N / self.fs)) * 60

        if len(zero_index) <= 1:
            RR2 = RR1
        else:
            time_span = 60 / RR1 / 2 * self.fs * QualityLevel
            zero_span = []
            for i in range(len(zero_index) - 1):
                zero_span.append(zero_index[i + 1] - zero_index[i])

            while min(zero_span) < time_span:
                doubt_point = np.argmin(zero_span)
                zero_index.pop(doubt_point)
                zero_index.pop(doubt_point)
                if len(zero_index) <= 1:
                    break
                zero_span = []
                for i in range(len(zero_index) - 1):
                    zero_span.append(zero_index[i + 1] - zero_index[i])

            zero_number = len(zero_index)
            cw = zero_number
            RR2 = ((cw / 2) / (self.N / self.fs)) * 60

        return RR2

def calculate_rms(signal):
    # 计算均方根值 (RMS)
    rms = np.sqrt(np.mean(signal ** 2))
    return rms

def Signal_Strength_Comparison(Signal1, Signal2):
    SignalA=calculate_rms(Signal1)
    SignalB=calculate_rms(Signal2)
    if SignalA <= SignalB:
        Signal_max=Signal1
        Signal_min=Signal2
    else:
        Signal_max=Signal2
        Signal_min=Signal1
    return Signal_max, Signal_min

def Signal_Calculate_Correlation(Signal1, Signal2):
    Correlation_Score = calculate_correlation(Signal1, Signal2)
    if Correlation_Score > 0.85:
        Signal_output = (Signal1+Signal2)/2
    else:
        Signal_max, Signal_min = Signal_Strength_Comparison(Signal1, Signal2)
        Noise_signal = Signal_max - Signal_min
        Signal_output = spectral_subtraction(Noise_signal, Signal_max)
    return Signal_output

def sine_interpolation(x1, x2, y1, y2, n_points, direction='up'):
    if direction == 'up':  # 上升趋势（从波谷到波峰）
        return y1 + (y2 - y1) * np.sin(np.linspace(0, np.pi / 2, n_points))
    elif direction == 'down':  # 下降趋势（从波峰到波谷）
        return y2 + (y1 - y2) * np.sin(np.linspace(np.pi / 2, np.pi, n_points))

def Noise_Removal_Module(signal):
    # 创建索引数组（时间或索引）
    t = np.arange(len(signal))

    # 初始信号的波峰和波谷
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)

    signal_output = Self_Adaption_Noise_Removal_Module(signal)

    # # 绘制修正后的信号及波峰波谷
    # plt.figure(figsize=(10, 6))

    # # 绘制原始信号
    # plt.plot(t, signal, label="Source Signal", alpha=0.5)
    #
    # # 绘制修正后的信号
    # plt.plot(t, signal_output, label="Edited Signal")
    # # 修改标题、坐标轴和图例
    # plt.title("Noise_Removal_Module (Sine Interpolation)")
    # plt.xlabel("Frame")
    # plt.ylabel("Amplitude")
    # plt.legend(title="legend")
    # plt.grid(True)
    #
    # plt.show()

    return signal_output


def sine_wave_generation(updated_peaks, updated_troughs, signal_length):
    # 确定波峰和波谷的比例关系，假设波峰和波谷是交替的
    # 确定 k 为波峰和波谷中的较小者
    k = min(updated_peaks, updated_troughs)

    # 相位控制逻辑
    if updated_peaks == updated_troughs:
        start_phase = 0  # 从0度开始
        end_phase = 2 * k * np.pi  # 结束在 2kπ
    elif updated_peaks > updated_troughs:
        start_phase = 0  # 从0度开始
        end_phase = (2 * k + 1) * np.pi  # 结束在 (2k+1)π
    else:
        start_phase = np.pi  # 从π开始
        end_phase = 2 * (k + 1) * np.pi  # 结束在 2(k+1)π

    # 生成从 start_phase 到 end_phase 的正弦波
    t = np.linspace(start_phase, end_phase, signal_length)
    amplitude = 1  # 振幅
    new_signal = amplitude * np.sin(t)

    return new_signal


def Self_Adaption_Noise_Removal_Module(signal):
    def calculate_threshold(signal):
        # 检测波峰
        peaks, _ = find_peaks(signal)

        # 检测波谷（负信号的波峰就是原信号的波谷）
        troughs, _ = find_peaks(-signal)

        # 存储垂直距离的列表
        peak_trough_distances = []

        # 计算波峰到左右波谷的垂直距离
        for i, peak in enumerate(peaks):
            left_troughs = [t for t in troughs if t < peak]
            right_troughs = [t for t in troughs if t > peak]

            if left_troughs:
                left_trough = left_troughs[-1]
                left_distance = signal[peak] - signal[left_trough]
                peak_trough_distances.append(left_distance)

            if right_troughs:
                right_trough = right_troughs[0]
                right_distance = signal[peak] - signal[right_trough]
                peak_trough_distances.append(right_distance)

        # 如果垂直距离的数量足够多
        if len(peak_trough_distances) > 2:
            median = np.median(peak_trough_distances)
            mad = np.median(np.abs(peak_trough_distances - median))

            # 使用中位数绝对偏差法去除异常值
            filtered_distances = [d for d in peak_trough_distances if np.abs(d - median) / mad <= 3]
        else:
            filtered_distances = peak_trough_distances

        # 计算剩余数据的平均值的 1/2 作为阈值
        if filtered_distances:
            average_distance = np.mean(filtered_distances)
            threshold = average_distance / 2.5
            print(f"\n当前判断阈值为: {threshold:.2f}")
        else:
            threshold = None

        return threshold, peaks, troughs

    def detect_and_update_peaks_troughs(signal, peaks, troughs, threshold):
        updated_peaks = len(peaks)
        updated_troughs = len(troughs)
        consecutive_peak_count = 0  # 连续波峰计数
        trough_candidates = []  # 用于记录触发条件的小于阈值的波谷坐标

        for i, peak in enumerate(peaks):
            # 查找相邻的左右波谷
            left_troughs = [t for t in troughs if t < peak]
            right_troughs = [t for t in troughs if t > peak]

            if left_troughs and right_troughs:
                left_trough = left_troughs[-1]  # 最近的左波谷
                right_trough = right_troughs[0]  # 最近的右波谷

                # 计算波峰到左右波谷的垂直距离
                left_distance = signal[peak] - signal[left_trough]
                right_distance = signal[peak] - signal[right_trough]

                # 检查垂直距离是否小于阈值
                if left_distance < threshold or right_distance < threshold:
                    consecutive_peak_count += 1  # 增加连续波峰计数

                    # 记录触发了小于阈值的波谷坐标
                    if left_distance < threshold:
                        trough_candidates.append(left_trough)
                    if right_distance < threshold:
                        trough_candidates.append(right_trough)

                    # 更新波峰和波谷数量
                    updated_peaks -= 1
                    updated_troughs -= 1

            # 单独处理第一个波峰（只有右侧波谷）
            elif i == 0 and right_troughs:
                right_trough = right_troughs[0]  # 第一个波峰只有右侧波谷
                right_distance = signal[peak] - signal[right_trough]
                if right_distance < threshold:  # 检查右侧波谷的距离是否小于阈值
                    updated_peaks -= 1
                    updated_troughs -= 1

            # 单独处理最后一个波峰（只有左侧波谷）
            elif i == len(peaks) - 1 and left_troughs:
                left_trough = left_troughs[-1]  # 最后一个波峰只有左侧波谷
                left_distance = signal[peak] - signal[left_trough]
                if left_distance < threshold:  # 检查左侧波谷的距离是否小于阈值
                    updated_peaks -= 1
                    updated_troughs -= 1

        # 二次检测触发恢复机制
        if consecutive_peak_count >= 2:  # 如果有两个或更多连续波峰
            # 查找是否有共享的波谷
            unique_troughs = set(trough_candidates)  # 获取唯一的波谷坐标
            for trough in unique_troughs:
                if trough_candidates.count(trough) > 1:  # 检查是否有波谷被多个波峰共享
                    # 如果找到共享波谷，触发恢复机制
                    updated_peaks += 1
                    updated_troughs += 1
                    break  # 找到一个共享的波谷后即可退出

        return updated_peaks, updated_troughs
    # 记录初始信号的波峰和波谷数量
    original_peaks, original_troughs = calculate_threshold(signal)[1:]

    # 计算初始阈值
    threshold, peaks, troughs = calculate_threshold(signal)
    if threshold is None:
        return signal  # 如果没有阈值，直接返回原信号

    # 检测并更新波峰和波谷数量
    updated_peaks, updated_troughs = detect_and_update_peaks_troughs(signal, peaks, troughs, threshold)

    # 根据更新的波峰和波谷数量生成正弦波形，并与原始信号对齐
    new_signal = sine_wave_generation(updated_peaks, updated_troughs, len(signal))  # 传递波峰和波谷数量

    return new_signal
def normalize_data(data):
    """对数据进行标准化处理"""
    data_max = np.max(data)
    data_min = np.min(data)
    normalized_data = (data - data_min) / (data_max - data_min) - 0.5
    return normalized_data

def calculate_vertical_distance(point1, point2):
    """计算两个3D点之间在y轴方向的垂直距离"""
    return abs(point1[1] - point2[1])

def calculate_average_distance(distance1, distance2):
    """计算三个输入的PSNR归一化值并根据权重叠加"""

    return (distance2+distance1)/2


def calculate_correlation(signal1, signal2):
    """
    计算两个信号的相关系数

    参数:
    signal1 (numpy array): 第一个信号
    signal2 (numpy array): 第二个信号

    返回:
    float: 相关系数
    """
    assert len(signal1) == len(signal2), "信号长度必须相同"

    correlation = np.corrcoef(signal1, signal2)[0, 1]
    return correlation

def low_pass_filter(signal, cutoff, fs):
    """应用低通滤波器"""
    # 进行傅里叶变换
    fft_signal = fft(signal)
    # 生成频率轴
    freqs = fftfreq(len(signal), 1 / fs)

    # 构建掩码，只保留低于 cutoff 的频率成分
    mask = np.abs(freqs) < cutoff
    filtered_fft = fft_signal * mask

    # 逆傅里叶变换恢复时域信号
    filtered_signal = ifft(filtered_fft)

    return np.real(filtered_signal)

def spectral_subtraction(noisy_signal, noise_estimation, alpha=0.3):
    # 对带噪声的信号和噪声估计进行傅里叶变换
    noisy_signal_fft = fft(noisy_signal)
    noise_fft = fft(noise_estimation)

    # 计算幅值
    noisy_magnitude = np.abs(noisy_signal_fft)
    noise_magnitude = np.abs(noise_fft)

    # 进行谱减法
    clean_magnitude = noisy_magnitude - alpha * noise_magnitude
    clean_magnitude = np.maximum(clean_magnitude, 0)  # 确保没有负值

    # 保持相位不变
    clean_signal_fft = clean_magnitude * np.exp(1j * np.angle(noisy_signal_fft))

    # 逆傅里叶变换回到时域
    clean_signal = ifft(clean_signal_fft)

    return np.real(clean_signal)

def npy_to_excel(npy_file_path, excel_file_path):
    """将 npy 数据转换为 Excel 文件，并计算关键点之间的距离"""
    # 读取 npy 文件
    data = np.load(npy_file_path, allow_pickle=True)

    # 初始化列表用于存储所有帧的数据
    rows = []

    # 遍历每帧的数据
    for frame_idx, keypoints in enumerate(data):
        hip = keypoints[0]  # hip
        right_hip = keypoints[1]  # right_hip
        left_hip = keypoints[2]  # left_hip
        left_shoulder = keypoints[4]  # left_shoulder
        right_shoulder = keypoints[5]  # right_shoulder

        # 计算距离
        LS2H = calculate_vertical_distance(left_shoulder, hip)  # 左肩到髋的距离
        RS2H = calculate_vertical_distance(right_shoulder, hip)  # 右肩到髋的距离
        average_distance = calculate_average_distance(LS2H, RS2H)

        # 将四个点的 (x, y, z) 坐标格式化为字符串
        frame_data = [f"{point[0]},{point[1]},{point[2]}" for point in keypoints]

        # 将帧编号、关键点数据和计算出的距离添加到每一行
        rows.append([frame_idx] + frame_data + [LS2H, RS2H, average_distance])

    # 定义列名，frame_id 表示帧的编号，接下来是六个点的名称和计算的距离列
    columns = ['frame_id', 'hip', 'right_hip', 'left_hip', 'spine', 'left_shoulder', 'right_shoulder', 'LS2H', 'RS2H',
               'average_distance']

    # 创建 DataFrame 并将数据写入 Excel 文件
    df = pd.DataFrame(rows, columns=columns)

    # 添加 Diff_LS2H 和 Diff_RS2H 列，并执行差值判断逻辑
    df['Diff_LS2H'] = 0
    df['Diff_RS2H'] = 0

    # 从第二行开始计算差值，并更新 Diff 列
    for i in range(1, len(df)):
        df.loc[i, 'Diff_LS2H'] = -1 if df.loc[i, 'LS2H'] - df.loc[i - 1, 'LS2H'] < 0 else (1 if df.loc[i, 'LS2H'] - df.loc[i - 1, 'LS2H'] > 0 else 0)
        df.loc[i, 'Diff_RS2H'] = -1 if df.loc[i, 'RS2H'] - df.loc[i - 1, 'RS2H'] < 0 else (1 if df.loc[i, 'RS2H'] - df.loc[i - 1, 'RS2H'] > 0 else 0)

    # 生成 Diff_Post_Process 列
    def post_process(row):
        if row['Diff_LS2H'] == -1 or row['Diff_RS2H'] == -1:
            return -1
        elif row['Diff_LS2H'] == 1 and row['Diff_RS2H'] == 1:
            return 1
        else:
            return 0

    df['Diff_Post_Process'] = df.apply(post_process, axis=1)

    # 将数据写入 Excel 文件
    df.to_excel(excel_file_path, index=False)

    print(f"数据已成功转换并保存到 {excel_file_path}")


def plot_waveform(optical_signal,excel_file_path,output_path):
    """从 Excel 文件读取数据并绘制 LS2H, RS2H 和 spine2hip 的波形"""
    # 读取 Excel 文件
    df = pd.read_excel(excel_file_path)

    # 提取帧编号和距离数据
    frames = df['frame_id']
    LS2H = df['LS2H'].to_numpy()
    RS2H = df['RS2H'].to_numpy()

    # 读取 Diff_LS2H, Diff_RS2H, Diff_Post_Process 数据
    Diff_LS2H = df['Diff_LS2H'].to_numpy()
    Diff_RS2H = df['Diff_RS2H'].to_numpy()
    Diff_Post_Process = df['Diff_Post_Process'].to_numpy()

    # 计算累积信号
    cumulative_LS2H = np.cumsum(Diff_LS2H)  # 累加 Diff_LS2H
    cumulative_RS2H = np.cumsum(Diff_RS2H)  # 累加 Diff_RS2H
    cumulative_Post_Process = np.cumsum(Diff_Post_Process)  # 累加 Diff_Post_Process

    Nom_cumulative_LS2H = normalize_data(apply_filter(cumulative_LS2H, fs=30))
    Nom_cumulative_RS2H = normalize_data(apply_filter(cumulative_RS2H, fs=30))
    Nom_cumulative_Post_Process = normalize_data(apply_filter(cumulative_Post_Process, fs=30))
    Nom_cumulative_Post_Process_0 = Noise_Removal_Module(Nom_cumulative_Post_Process)

    # Diff_signal
    print('-------------------------------------')
    Diff_LS2H_RR = RespirationRateCalculator(Nom_cumulative_LS2H, 30, len(Nom_cumulative_LS2H))
    Diff_LS2H_RR_NFCP = Diff_LS2H_RR.NegativeFeedbackCrossoverPointMethod()
    print(f'Calculated Breathing Rate(LS2H)(NFCP) from Diff: {Diff_LS2H_RR_NFCP:.2f} breaths per minute')

    Diff_RS2H_RR = RespirationRateCalculator(Nom_cumulative_RS2H, 30, len(Nom_cumulative_RS2H))
    Diff_RS2H_RR_NFCP = Diff_RS2H_RR.NegativeFeedbackCrossoverPointMethod()
    print(f'Calculated Breathing Rate(RS2H)(NFCP) from Diff: {Diff_RS2H_RR_NFCP:.2f} breaths per minute')

    # 距离信号
    diff_filtered_LS2H = apply_filter(LS2H, fs=30)
    diff_filtered_RS2H = apply_filter(RS2H, fs=30)
    LS2H_Norm = normalize_data(diff_filtered_LS2H)

    RS2H_Norm = normalize_data(diff_filtered_RS2H)


    LS2H_1 = LS2H_Norm-optical_signal
    RS2H_1 = RS2H_Norm-optical_signal

    Source_signal = Signal_Calculate_Correlation(LS2H_Norm, RS2H_Norm)
    Source_signal_post_process = Noise_Removal_Module(Source_signal)


    rescuve = Signal_Calculate_Correlation(LS2H_1, RS2H_1)
    output_signal= Noise_Removal_Module(rescuve)

    # 处理信号呼吸率
    print('-------------------------------------')
    rr_calculator_rescuve = RespirationRateCalculator(rescuve, 30, len(rescuve))
    breathing_rate_rescuve = rr_calculator_rescuve.NegativeFeedbackCrossoverPointMethod()
    breathing_rate_rescuve_CrossingPoint = rr_calculator_rescuve.CrossingPoint()
    print(f'Calculated Breathing Rate(NFCP) from Output_Signal: {breathing_rate_rescuve:.2f} breaths per minute')
    print(f'Calculated Breathing Rate(CP) from Output_Signal: {breathing_rate_rescuve_CrossingPoint:.2f} breaths per minute')

    rr_calculator_output_signal = RespirationRateCalculator(output_signal, 30, len(output_signal))
    calculator_output_signal_CP = rr_calculator_output_signal.CrossingPoint()
    print(f'Calculated Breathing Rate(Noise_Removal_Module) from Output_Signal: {calculator_output_signal_CP:.2f} breaths per minute')

    print('-------------------------------------')
    # 源姿态信号(非差分)
    rr_calculator_Source_signal = RespirationRateCalculator(Source_signal, 30, len(Source_signal))
    breathing_rate_Source_signal_NFCP = rr_calculator_Source_signal.NegativeFeedbackCrossoverPointMethod()
    breathing_rate_Source_signal_CrossingPoint = rr_calculator_Source_signal.CrossingPoint()
    print(f'Calculated Breathing Rate(NFCP) from Source_signal: {breathing_rate_Source_signal_NFCP:.2f} breaths per minute')
    print(f'Calculated Breathing Rate(CP) from Source_signal: {breathing_rate_Source_signal_CrossingPoint:.2f} breaths per minute')

    rr_calculator_Source_signal_Post_Process = RespirationRateCalculator(Source_signal_post_process, 30, len(Source_signal_post_process))
    breathing_rate_Source_signal_Post_Process_CrossingPoint = rr_calculator_Source_signal_Post_Process.CrossingPoint()
    print(f'Calculated Breathing Rate(Noise_Removal_Module) from Source_signal_Post_Process: {breathing_rate_Source_signal_Post_Process_CrossingPoint:.2f} breaths per minute')



    # 创建绘图
    # plt.figure(figsize=(10, 6))
    # plt.plot(frames, RS2H_Norm, label='RS2H', color='green')
    # plt.plot(frames, LS2H_Norm, label='LS2H', color='red')
    # plt.plot(frames, rescuve, label='Source_signal', color='blue')
    # plt.title('Respiratory Rate', fontsize=16)
    # plt.xlabel('Frame', fontsize=12)
    # plt.ylabel('Distance', fontsize=12)
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    output_image_path = os.path.join(output_path, 'Diff_signal.png')  # 保存图像的路径

    plt.figure(figsize=(10, 6))
    plt.plot(frames, Nom_cumulative_LS2H, label='Diff_LS2H', color='green')
    plt.plot(frames, Nom_cumulative_RS2H, label='Diff_RS2H', color='red')
    plt.title('Diff_Respiratory Rate', fontsize=16)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image_path, dpi=300)  # 通过 dpi 参数设置图像分辨率
    plt.close()  # 关闭图像以释放内存


    return Diff_LS2H_RR_NFCP, Diff_RS2H_RR_NFCP



# def apply_filter(signal_data, filter_order=3, low_pass=5/60, high_pass=50/60, fs=30, filter_type='band'):
#     """应用Butterworth滤波器"""
#     b, a = signal.butter(filter_order, [2 * low_pass / fs, 2 * high_pass / fs], btype=filter_type)
#     return signal.filtfilt(b, a, signal_data)

def apply_filter(signal_data, filter_order=3, low_pass=10/60, high_pass=50/60, fs=30, filter_type='band'):
    # 设计Butterworth滤波器
    b, a = signal.butter(filter_order, [2 * low_pass / fs, 2 * high_pass / fs], btype=filter_type)
    pad_len = len(signal_data) // 2
    padded_signal = np.pad(signal_data, pad_len, mode='reflect')
    filtered_padded_signal = signal.filtfilt(b, a, padded_signal)
    filtered_signal = filtered_padded_signal[pad_len:-pad_len]
    return filtered_signal


def normalize_data(data):
    """对数据进行标准化处理"""
    data_max = np.max(data)
    data_min = np.min(data)
    normalized_data = (data - data_min) / (data_max - data_min) - 0.5
    return normalized_data

# 示例调用
def motion_signal_extract(optical_signal,npy_path,excel_file_path,RRImage_path):
   npy_file_path = npy_path
   excel_file_path = excel_file_path
   npy_to_excel(npy_file_path, excel_file_path)
   Diff_LS2H_RR_NFCP, Diff_RS2H_RR_NFCP = plot_waveform(optical_signal, excel_file_path,RRImage_path)
   return Diff_LS2H_RR_NFCP, Diff_RS2H_RR_NFCP,
