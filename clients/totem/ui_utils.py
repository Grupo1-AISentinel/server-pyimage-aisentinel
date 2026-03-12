import cv2
import numpy as np

def draw_futuristic_box(img, x1, y1, x2, y2, color, text=None, text_size=0.55):
    """
    Dibuja un bounding box futurista estilo HUD (esquinas resaltadas)
    y un banner de texto semitransparente moderno con tipografía ajustada.
    """
    # Escala dinámica basada en resolución (referencia 1080p -> altura 1080 -> scale ~1.0)
    h_img, w_img = img.shape[:2]
    scale = max(1.0, min(h_img, w_img) / 800.0)
    
    thickness = max(1, int(2 * scale))
    corner_length = max(15, int(15 * scale))
    text_size = max(0.4, text_size * scale * 0.8)
    font_thickness = max(1, int(1 * scale))

    # 1. Caja tenue semitransparente o punteada (aquí hacemos un rectángulo base delgado y oscuro)
    cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0]*0.4), int(color[1]*0.4), int(color[2]*0.4)), 1)
    
    # 2. Esquinas (HUD style)
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, thickness)

    # 3. Text Banner elegante y semitransparente
    if text:
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, text_size, 1)
        
        # Coordenadas del banner
        bx1 = x1
        by1 = max(0, y1 - th - 10)
        bx2 = x1 + tw + 10
        by2 = max(0, y1)

        # Usar overlay para transparencia alfa
        overlay = img.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 25), cv2.FILLED)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # LÍNEA DE ACENTO SUPERIOR para el banner
        cv2.line(img, (bx1, by1), (bx2, by1), color, thickness)

        # Texto nítido
        cv2.putText(img, text, (bx1 + int(5*scale), by1 + th + int(4*scale)), font, text_size, color, font_thickness, cv2.LINE_AA)
