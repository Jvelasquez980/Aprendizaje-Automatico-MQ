import os
import shutil
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

class ImageValidator:
    """Valida y filtra imágenes de cancer de sangre"""
    
    def __init__(self, data_dir='data', min_width=50, min_height=50):
        self.data_dir = data_dir
        self.min_width = min_width
        self.min_height = min_height
        self.results = {
            'total_processed': 0,
            'valid_images': 0,
            'corrupted': [],
            'too_small': [],
            'unsupported_format': [],
            'categories': {}
        }
        
    def validate_image(self, image_path):
        """
        Valida una imagen individual
        Retorna: (is_valid, reason)
        """
        try:
            with Image.open(image_path) as img:
                # Verificar que es image mode válido
                if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    return False, 'unsupported_format'
                
                # Verificar dimensiones mínimas
                width, height = img.size
                if width < self.min_width or height < self.min_height:
                    return False, 'too_small'
                
                return True, 'valid'
                
        except Image.UnidentifiedImageError:
            return False, 'corrupted'
        except Exception as e:
            return False, 'corrupted'
    
    def process_directory(self, delete_invalid=False):
        """
        Procesa todas las imágenes en el directorio de datos
        
        Args:
            delete_invalid: Si True, elimina imágenes inválidas
                           Si False, solo reporta
        """
        data_path = Path(self.data_dir)
        
        if not data_path.exists():
            print(f"❌ El directorio '{self.data_dir}' no existe")
            return self.results
        
        print(f"🔍 Validando imágenes en {self.data_dir}/")
        print(f"   Dimensión mínima: {self.min_width}x{self.min_height}")
        print(f"   Modo: {'ELIMINAR archivos inválidos' if delete_invalid else 'Solo reportar'}\n")
        
        # Extensiones de imagen soportadas
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # Procesar cada categoría
        for category_path in data_path.iterdir():
            if not category_path.is_dir():
                continue
            
            category_name = category_path.name
            self.results['categories'][category_name] = {
                'valid': 0,
                'corrupted': 0,
                'too_small': 0,
                'unsupported_format': 0,
                'total': 0
            }
            
            print(f"📁 {category_name}")
            
            # Procesar imágenes en la categoría
            for image_file in category_path.iterdir():
                if image_file.suffix.lower() not in image_extensions:
                    continue
                
                self.results['total_processed'] += 1
                self.results['categories'][category_name]['total'] += 1
                
                is_valid, reason = self.validate_image(image_file)
                
                if is_valid:
                    self.results['valid_images'] += 1
                    self.results['categories'][category_name]['valid'] += 1
                else:
                    # Guardar registro del problema
                    self.results[reason].append(str(image_file))
                    self.results['categories'][category_name][reason] += 1
                    
                    # Eliminar archivo si se indica
                    if delete_invalid:
                        try:
                            image_file.unlink()
                            print(f"   ✓ Eliminada: {image_file.name} ({reason})")
                        except Exception as e:
                            print(f"   ⚠ Error al eliminar {image_file.name}: {e}")
            
            # Resumen por categoría
            cat_stats = self.results['categories'][category_name]
            print(f"   ✓ Válidas: {cat_stats['valid']}/{cat_stats['total']}")
            if cat_stats['corrupted'] > 0:
                print(f"   ✗ Corrupted: {cat_stats['corrupted']}")
            if cat_stats['too_small'] > 0:
                print(f"   ⚠ Muy pequeñas: {cat_stats['too_small']}")
            if cat_stats['unsupported_format'] > 0:
                print(f"   ⚠ Formato no soportado: {cat_stats['unsupported_format']}")
            print()
        
        return self.results
    
    def print_summary(self):
        """Imprime resumen de la validación"""
        print("=" * 60)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("=" * 60)
        print(f"Total procesadas: {self.results['total_processed']}")
        print(f"✓ Imágenes válidas: {self.results['valid_images']}")
        print(f"✗ Imágenes corrupted: {len(self.results['corrupted'])}")
        print(f"⚠ Imágenes muy pequeñas: {len(self.results['too_small'])}")
        print(f"⚠ Formato no soportado: {len(self.results['unsupported_format'])}")
        print(f"\nPorcentaje de validez: {100*self.results['valid_images']/max(1,self.results['total_processed']):.1f}%")
        print("=" * 60)
    
    def save_report(self, report_file='validation_report.json'):
        """Guarda reporte detallado en JSON"""
        # Convertir sets a listas para JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'min_dimensions': f"{self.min_width}x{self.min_height}",
            'summary': {
                'total_processed': self.results['total_processed'],
                'valid_images': self.results['valid_images'],
                'corrupted_count': len(self.results['corrupted']),
                'too_small_count': len(self.results['too_small']),
                'unsupported_format_count': len(self.results['unsupported_format']),
            },
            'categories': self.results['categories'],
            'invalid_files': {
                'corrupted': self.results['corrupted'],
                'too_small': self.results['too_small'],
                'unsupported_format': self.results['unsupported_format']
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Reporte guardado en: {report_file}")


if __name__ == '__main__':
    # Configuración
    MIN_WIDTH = 1024    # Ancho mínimo en píxeles
    MIN_HEIGHT = 728   # Alto mínimo en píxeles
    DELETE_INVALID = False  # Cambiar a True para eliminar archivos inválidos
    
    # Crear validador
    validator = ImageValidator(
        data_dir='data/Blood cell Cancer [ALL]',
        min_width=MIN_WIDTH,
        min_height=MIN_HEIGHT
    )
    
    # Procesar imágenes
    results = validator.process_directory(delete_invalid=DELETE_INVALID)
    
    # Mostrar resumen
    validator.print_summary()
    
    # Guardar reporte
    validator.save_report('Manejo_de_datos/reportes/validation_report.json')
    
    print("\n💡 Para ELIMINAR imágenes inválidas, cambiar DELETE_INVALID a True en el script")
