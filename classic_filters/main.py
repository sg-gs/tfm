import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
from scipy.signal import wiener, medfilt2d
from scipy.fft import fft2, ifft2, fftfreq
from skimage.metrics import structural_similarity as ssim
import pywt
import os
from datetime import datetime

class StokesImageFilters:    
    def __init__(self, fits_file):
        self.fits_file = fits_file
        self.original_data = None
        self.stokes_cube = None
        self.header = None
        self.load_fits_data()
        
    def load_fits_data(self):
        print(f"Cargando {self.fits_file}...")
        
        with fits.open(self.fits_file) as hdul:
            data = hdul[0].data.astype(np.float32)
            self.header = hdul[0].header
        
        # Asumir formato (6, 4, H, W) y tomar primer elemento
        if len(data.shape) == 4 and data.shape[0] == 6 and data.shape[1] == 4:
            self.stokes_cube = np.transpose(data[0], (1, 2, 0))  # (H, W, 4)
        else:
            raise ValueError(f"Formato inesperado: {data.shape}")
        
        self.original_data = self.stokes_cube.copy()
        print(f"Datos cargados: {self.stokes_cube.shape}")
    
    def gaussian_filter(self, sigma=1.0):
        print(f"Aplicando filtro Gaussiano (sigma={sigma})...")
        
        filtered_cube = np.zeros_like(self.stokes_cube)
        
        for i in range(4):
            filtered_cube[:, :, i] = ndimage.gaussian_filter(
                self.stokes_cube[:, :, i], 
                sigma=sigma
            )
        
        return filtered_cube
    
    def wiener_filter(self, noise_var=None):
        print(f"Aplicando filtro de Wiener...")
        
        filtered_cube = np.zeros_like(self.stokes_cube)
        
        for i in range(4):
            stokes_channel = self.stokes_cube[:, :, i]
            
            # Estimar varianza del ruido si no se proporciona
            if noise_var is None:
                # Usar la varianza de las diferencias como estimación del ruido
                diff_h = np.diff(stokes_channel, axis=0)
                diff_v = np.diff(stokes_channel, axis=1)
                estimated_noise_var = (np.var(diff_h) + np.var(diff_v)) / 2
            else:
                estimated_noise_var = noise_var
            
            # Aplicar filtro de Wiener
            filtered_cube[:, :, i] = wiener(stokes_channel, noise=estimated_noise_var)
        
        return filtered_cube
    
    def median_filter(self, kernel_size=3):
        print(f"Aplicando filtro de mediana (kernel={kernel_size}x{kernel_size})...")
        
        filtered_cube = np.zeros_like(self.stokes_cube)
        
        for i in range(4):
            filtered_cube[:, :, i] = medfilt2d(
                self.stokes_cube[:, :, i], 
                kernel_size=kernel_size
            )
        
        return filtered_cube
    
    def wavelet_filter(self, wavelet='db4', sigma=None, mode='soft'):
        print(f"Aplicando filtro de Wavelets (wavelet={wavelet}, mode={mode})...")
        
        filtered_cube = np.zeros_like(self.stokes_cube)
        
        for i in range(4):
            stokes_channel = self.stokes_cube[:, :, i]
            
            # Descomposición wavelet
            coeffs = pywt.wavedec2(stokes_channel, wavelet, mode='symmetric')
            
            # Estimar nivel de ruido si no se proporciona
            if sigma is None:
                # Usar la mediana de los coeficientes de detalle del primer nivel
                sigma_est = np.median(np.abs(coeffs[-1][-1])) / 0.6745
            else:
                sigma_est = sigma
            
            # Calcular threshold
            threshold = sigma_est * np.sqrt(2 * np.log(stokes_channel.size))
            
            # Aplicar thresholding a los coeficientes de detalle
            coeffs_thresh = [coeffs[0]]  # Mantener aproximación
            for level_coeffs in coeffs[1:]:
                thresh_level = tuple(pywt.threshold(detail, threshold, mode=mode) for detail in level_coeffs)
                coeffs_thresh.append(thresh_level)
                        
            # Reconstruir imagen
            filtered_cube[:, :, i] = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
        
        return filtered_cube
    
    def fourier_filter(self, cutoff_freq=0.3, filter_type='lowpass'):
        print(f"Aplicando filtro de Fourier ({filter_type}, cutoff={cutoff_freq})...")
        
        filtered_cube = np.zeros_like(self.stokes_cube)
        
        for i in range(4):
            stokes_channel = self.stokes_cube[:, :, i]
            h, w = stokes_channel.shape
            
            # Transformada de Fourier
            fft_image = fft2(stokes_channel)
            fft_shifted = np.fft.fftshift(fft_image)
            
            # Crear máscara de frecuencias
            center_h, center_w = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            
            # Distancia desde el centro
            distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            distance_normalized = distance / (min(h, w) / 2)
            
            # Crear filtro
            if filter_type == 'lowpass':
                # Filtro paso bajo con transición suave
                mask = 1 / (1 + (distance_normalized / cutoff_freq)**10)
            elif filter_type == 'highpass':
                # Filtro paso alto
                mask = 1 - 1 / (1 + (distance_normalized / cutoff_freq)**10)
            elif filter_type == 'bandpass':
                # Filtro paso banda
                low_cutoff = cutoff_freq * 0.5
                high_cutoff = cutoff_freq * 1.5
                mask_low = 1 / (1 + (distance_normalized / low_cutoff)**10)
                mask_high = 1 - 1 / (1 + (distance_normalized / high_cutoff)**10)
                mask = mask_high * mask_low
            
            # Aplicar filtro
            fft_filtered = fft_shifted * mask
            fft_ishifted = np.fft.ifftshift(fft_filtered)
            
            # Transformada inversa
            filtered_image = np.real(ifft2(fft_ishifted))
            filtered_cube[:, :, i] = filtered_image
        
        return filtered_cube
    
    def validate_stokes_physics(self, stokes_cube):
        I = stokes_cube[..., 0]
        Q = stokes_cube[..., 1] 
        U = stokes_cube[..., 2]
        V = stokes_cube[..., 3]
        
        pol_intensity = np.sqrt(Q**2 + U**2 + V**2)
        violations = np.sum(pol_intensity > I * 1.001)  # Misma tolerancia
        total_pixels = I.size
        compliance_rate = 1.0 - (violations / total_pixels)
        
        return {
            'violations': violations,
            'total_pixels': total_pixels, 
            'compliance_rate': compliance_rate
        }
    
    def calculate_metrics(self, original, filtered):
        metrics = {}
        stokes_names = ['I', 'Q', 'U', 'V']
        
        for i, name in enumerate(stokes_names):
            orig_channel = original[..., i]
            filt_channel = filtered[..., i]
            
            # MSE
            mse = np.mean((orig_channel - filt_channel)**2)
            
            # PSNR
            if mse > 0:
                psnr = 20 * np.log10(orig_channel.max() / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            try:
                ssim_value = ssim(orig_channel, filt_channel, 
                                data_range=orig_channel.max()-orig_channel.min())
            except:
                ssim_value = 0.0
            
            # Reducción de ruido (basada en varianza)
            orig_std = np.std(orig_channel)
            filt_std = np.std(filt_channel)
            noise_reduction = (orig_std - filt_std) / orig_std if orig_std > 0 else 0
            
            metrics[name] = {
                'MSE': mse,
                'PSNR': psnr,
                'SSIM': ssim_value,
                'noise_reduction': noise_reduction
            }
        
        return metrics
    
    def save_fits(self, filtered_data, filename, filter_name):
        # Transponer para formato FITS (C, H, W)
        fits_data = np.transpose(filtered_data, (2, 0, 1))
        
        # Crear HDU
        hdu = fits.PrimaryHDU(fits_data.astype(np.float32))
        
        # Copiar header original y añadir información del filtro
        if self.header:
            hdu.header = self.header.copy()
        
        hdu.header['FILTER'] = filter_name
        hdu.header['HISTORY'] = f'Applied {filter_name} filter'
        hdu.header['DATE'] = datetime.now().isoformat()
        
        hdu.writeto(filename, overwrite=True)
        print(f"Guardado: {filename}")
    
    def create_comparison_plot(self, results_dict, save_path='filter_comparison.png'):
        stokes_names = ['I', 'Q', 'U', 'V']
        filters = list(results_dict.keys())
        n_filters = len(filters)
        
        fig, axes = plt.subplots(n_filters + 1, 4, figsize=(20, 5 * (n_filters + 1)))
        
        # Primera fila: imagen original
        for j, stokes_name in enumerate(stokes_names):
            im = axes[0, j].imshow(
                self.original_data[..., j], 
                cmap='hot' if j == 0 else 'coolwarm',
                origin='lower'
            )
            axes[0, j].set_title(f'Original {stokes_name}', fontweight='bold')
            axes[0, j].axis('off')
            plt.colorbar(im, ax=axes[0, j], fraction=0.046)
        
        # Filas siguientes: filtros
        for i, (filter_name, filtered_data) in enumerate(results_dict.items()):
            for j, stokes_name in enumerate(stokes_names):
                im = axes[i + 1, j].imshow(
                    filtered_data[..., j], 
                    cmap='hot' if j == 0 else 'coolwarm',
                    origin='lower'
                )
                axes[i + 1, j].set_title(f'{filter_name} {stokes_name}', fontweight='bold')
                axes[i + 1, j].axis('off')
                plt.colorbar(im, ax=axes[i + 1, j], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Comparación guardada: {save_path}")
    
    def create_metrics_table(self, metrics_summary):
        print("\n" + "=" * 100)
        print("TABLA COMPARATIVA DE MÉTRICAS")
        print("=" * 100)
        
        stokes_names = ['I', 'Q', 'U', 'V']
        filters = list(metrics_summary.keys())
        
        for stokes in stokes_names:
            print(f"\nParámetro {stokes}:")
            print("-" * 80)
            print(f"{'Filtro':<12} {'MSE':<12} {'PSNR (dB)':<12} {'SSIM':<8} {'Reducción %':<12}")
            print("-" * 80)
            
            for filter_name in filters:
                metrics = metrics_summary[filter_name][stokes]
                print(f"{filter_name:<12} {metrics['MSE']:<12.6f} "
                      f"{metrics['PSNR']:<12.2f} {metrics['SSIM']:<8.4f} "
                      f"{metrics['noise_reduction']:<12.1%}")
        
        # Encontrar mejores métricas por parámetro
        print("\n" + "=" * 60)
        print("MEJOR FILTRO POR MÉTRICA Y PARÁMETRO")
        print("=" * 60)
        
        for stokes in stokes_names:
            print(f"\nParámetro {stokes}:")
            
            # Mejor MSE (menor es mejor)
            best_mse = min(filters, key=lambda f: metrics_summary[f][stokes]['MSE'])
            best_mse_val = metrics_summary[best_mse][stokes]['MSE']
            
            # Mejor PSNR (mayor es mejor)
            best_psnr = max(filters, key=lambda f: metrics_summary[f][stokes]['PSNR'])
            best_psnr_val = metrics_summary[best_psnr][stokes]['PSNR']
            
            # Mejor SSIM (mayor es mejor)
            best_ssim = max(filters, key=lambda f: metrics_summary[f][stokes]['SSIM'])
            best_ssim_val = metrics_summary[best_ssim][stokes]['SSIM']
            
            # Mejor reducción de ruido (mayor es mejor)
            best_noise = max(filters, key=lambda f: metrics_summary[f][stokes]['noise_reduction'])
            best_noise_val = metrics_summary[best_noise][stokes]['noise_reduction']
            
            print(f"  🥇 MSE: {best_mse} ({best_mse_val:.6f})")
            print(f"  🥇 PSNR: {best_psnr} ({best_psnr_val:.2f} dB)")
            print(f"  🥇 SSIM: {best_ssim} ({best_ssim_val:.4f})")
            print(f"  🥇 Reducción ruido: {best_noise} ({best_noise_val:.1%})")
    
    def run_all_filters(self, save_fits_files=True, save_comparison=True):
        results = {}
        metrics_summary = {}
        
        filter_configs = {
            'Gaussian': {'sigma': 1.0},
            'Wiener': {'noise_var': None},
            'Median': {'kernel_size': 3},
            'Wavelets': {'wavelet': 'db4', 'mode': 'soft'},
            'Fourier': {'cutoff_freq': 0.3, 'filter_type': 'lowpass'}
        }
        
        for filter_name, config in filter_configs.items():
            print(f"\n--- {filter_name} ---")
            
            if filter_name == 'Gaussian':
                filtered = self.gaussian_filter(**config)
            elif filter_name == 'Wiener':
                filtered = self.wiener_filter(**config)
            elif filter_name == 'Median':
                filtered = self.median_filter(**config)
            elif filter_name == 'Wavelets':
                filtered = self.wavelet_filter(**config)
            elif filter_name == 'Fourier':
                filtered = self.fourier_filter(**config)
            
            results[filter_name] = filtered
            
            # Calcular métricas
            metrics = self.calculate_metrics(self.original_data, filtered)
            metrics_summary[filter_name] = metrics
            
            # Validar física
            physics_stats = self.validate_stokes_physics(filtered)
            
            print(f"Cumplimiento físico: {physics_stats['compliance_rate']:.4f} "
                  f"({physics_stats['compliance_rate']*100:.2f}%)")
            
            # Guardar FITS si se solicita
            if save_fits_files:
                filename = f"{filter_name.lower()}_filtered_stokes.fits"
                self.save_fits(filtered, filename, filter_name)
        
        print("\n" + "=" * 60)
        print("RESUMEN DE MÉTRICAS")
        print("=" * 60)
        
        print(f"\n{filter_name}:")
        for filter_name, metrics in metrics_summary.items():
            print(f"\n{filter_name}:")
            for stokes, values in metrics.items():
                print(f"  {stokes}: MSE={values['MSE']:.6f}, "
                      f"PSNR={values['PSNR']:.2f} dB, "
                      f"SSIM={values['SSIM']:.4f}, "
                      f"Reducción ruido={values['noise_reduction']:.1%}")
        
        # Crear tabla resumen de métricas
        self.create_metrics_table(metrics_summary)
        
        # Crear comparación visual
        if save_comparison:
            self.create_comparison_plot(results)
        
        return results, metrics_summary


def main():
    """Función principal"""
    fits_file = '../data.fits'
    
    if not os.path.exists(fits_file):
        print(f"No se encuentra el archivo: {fits_file}")
        print("Por favor, ajusta la variable 'fits_file' con la ruta correcta")
        return
    
    try:
        # Crear instancia del filtrador
        filter_processor = StokesImageFilters(fits_file)
        
        # Ejecutar todos los filtros
        results, metrics = filter_processor.run_all_filters(
            save_fits_files=True,
            save_comparison=True
        )
        
        return filter_processor, results, metrics
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    filter_processor, results, metrics = main()