import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u 

# Buscar datos HMI de intensidad continua
results = Fido.search(
    a.Time('2019/05/15 12:00:00', '2019/05/15 12:30:00'),
    a.Instrument('HMI'),
    a.Physobs('intensity'),
    a.Sample(1 * u.minute)  # Usar 1 minuto como intervalo de muestreo
)

# Descargar los resultados
downloaded_files = Fido.fetch(results)

# Ver la ruta del archivo descargado
print(f"Archivo descargado: {downloaded_files[0]}")

# Cargar como mapa de SunPy
hmi_map = sunpy.map.Map(downloaded_files[0])

# Visualizar la imagen
plt.figure(figsize=(10, 10))
hmi_map.plot()
plt.colorbar()
plt.title(f"HMI Intensitygram {hmi_map.date}")
plt.show()

# Examinar el formato del archivo y algunos metadatos
with fits.open(downloaded_files[0]) as hdul:
    print(f"Formato del archivo: {hdul.info()}")
    print("\nAlgunos metadatos:")
    for key in ['DATE-OBS', 'TELESCOP', 'INSTRUME', 'WAVELNTH', 'QUALITY']:
        if key in hdul[1].header:
            print(f"{key}: {hdul[1].header[key]}")
    
    # Información sobre los datos
    print(f"\nDimensiones: {hdul[1].data.shape}")
    print(f"Tipo de datos: {hdul[1].data.dtype}")
