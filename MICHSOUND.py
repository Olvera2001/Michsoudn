import numpy as np
import matplotlib.pyplot as graficar
from scipy import fft, signal
import scipy
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq

# Configuración de las gráficas
graficar.rcParams['figure.figsize'] = [8, 6]
graficar.rcParams['figure.dpi'] = 140

# Leer los archivos .WAV
# Fs es la frecuencia de muestreo del archivo
Fs, cancion = read("D:\MICHSOUND\grabar audio\grabacion_1.wav")
# Analizar uno solo de los canales
tonada = cancion[:,0]


# Revisar un pedazo de la señal en el tiempo
# Vector que va de la Fs hasta Fs * 1.3 del tipo entero
time_to_plot = np.arange(Fs * 1, Fs * 1.3, dtype=int)

# Analizamos la canción en frecuencia
N = len(tonada)
fftcancion = fft(tonada)
transform_y = 2.0 / N * np.abs(fftcancion[0:N//2])
transform_x = fftfreq(N, 1 / Fs)[:N//2]

graficar.subplot(2, 1, 1)
graficar.plot(time_to_plot, tonada[time_to_plot])
graficar.title("Señal sonido en el tiempo")
graficar.xlabel("Índice de Tiempo")
graficar.ylabel("Magnitud")
graficar.subplot(2, 1, 2)
graficar.plot(transform_x, transform_y)
graficar.xlabel("Frecuencia (Hz)")
graficar.xlim(0,2000)
graficar.grid()
graficar.show()


# TODO: Analizar como se encuentran los picos
all_peaks, props = signal.find_peaks(transform_y)
picos, props = signal.find_peaks(transform_y, prominence=0, distance=10000)
numerosdepicos = 12


largest_peaks_indices = np.argpartition(props["prominences"], -numerosdepicos)[-numerosdepicos:]
largest_peaks = picos[largest_peaks_indices]

graficar.plot(transform_x, transform_y, label="Espectro")
graficar.scatter(transform_x[largest_peaks], transform_y[largest_peaks], color="r", zorder=10, label="Constrained Peaks")
graficar.xlim(0, 3000)
graficar.show()


# parametros
window_length_seconds = 3
window_length_samples = int(window_length_seconds * Fs)
window_length_samples += window_length_samples % 2


frequencies, times, stft = signal.stft(
    tonada, Fs, nperseg=window_length_samples,
    nfft=window_length_samples, return_onesided=True
)

stft.shape


constellation_map = []

for time_idx, window in enumerate(stft.T):
  
    spectrum = abs(window)
    
    peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)

   
    n_peaks = 5
   
    largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
    for peak in peaks[largest_peaks]:
        frequency = frequencies[peak]
        constellation_map.append([time_idx, frequency])


graficar.scatter(*zip(*constellation_map));


# Número de puntos
N = 600
# Muestreo a 1/800 s
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
# f(t) = A*sin(w*t)         donde w = 2pi*f  
# y = sin(5 * 2pi * x) + 0.75*sin(10 * 2pi * x)
y = np.sin(5.0 * 2.0*np.pi*x) + 0.75*np.sin(10.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]