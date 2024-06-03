from ExtraccionFrames import process_all_videos
from PromediarFrames import process_all_averaging
from ConversionCSV import process_all_to_csv
from EntrenarCNN import train_model

def main():
    # Extraer frames de los videos
    process_all_videos('GIT-Mask/GIT-Mask/video-outputs', 'GIT-Mask/GIT-Mask/frames', 5)
    
    # Promediar los frames extraídos
    process_all_averaging('GIT-Mask/GIT-Mask/frames', 'GIT-Mask/GIT-Mask/averaged')
    
    # Convertir los frames promediados a archivos CSV
    process_all_to_csv('GIT-Mask/GIT-Mask/averaged', 'GIT-Mask/GIT-Mask/csv')
    
    # Entrenar la red CNN con los datos CSV
    train_model('GIT-Mask/GIT-Mask/csv')
    
    print("Proceso completo: Extracción, Promedio, Conversión y Entrenamiento de la CNN.")

if __name__ == '__main__':
    main()
