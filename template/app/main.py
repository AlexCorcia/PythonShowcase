import numpy as np
import time

from app.utils import (
    limpiar_consola,
    crear_barra,
    mensaje_estado,
    LOGO
)

def main():
    velocidades = np.random.randint(1, 2, size=100)

    for i in range(101):
        limpiar_consola()

        print("TEMPLATE PYTHON LOADING\n")
        print(f"[{crear_barra(i)}] {i}%\n")
        print(mensaje_estado(i))

        time.sleep(velocidades[i-1] * 0.0001 if i > 0 else 0.0005)

    limpiar_consola()
    print(LOGO)
    print("Proyecto listo para usar\n")

if __name__ == "__main__":
    main()
