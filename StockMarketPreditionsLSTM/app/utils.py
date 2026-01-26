import os

LOGO = r"""
████████╗███████╗███╗   ███╗██████╗ ██╗      █████╗ ████████╗███████╗
╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝
   ██║   █████╗  ██╔████╔██║██████╔╝██║     ███████║   ██║   █████╗  
   ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  
   ██║   ███████╗██║ ╚═╝ ██║██║     ███████╗██║  ██║   ██║   ███████╗
   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""

def limpiar_consola():
    os.system("cls" if os.name == "nt" else "clear")

def crear_barra(porcentaje, tamaño=40):
    llenos = int((porcentaje / 100) * tamaño)
    vacios = tamaño - llenos
    return "█" * llenos + "-" * vacios

def mensaje_estado(p):
    if p < 30:
        return "Inicializando módulos..."
    elif p < 60:
        return "Cargando dependencias..."
    elif p < 85:
        return "Optimizando rendimiento..."
    else:
        return "Finalizando..."
