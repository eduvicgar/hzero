# hzero (ES)

**hzero** es una librería de inferencia estadística enfocada en la implementación de contrastes de hipótesis, 
tanto paramétricos como no paramétricos. 
Está pensada como una herramienta educativa y experimental para explorar el comportamiento de diferentes tests 
estadísticos bajo diversas condiciones.

## 🚧 Estado del proyecto

Este proyecto está actualmente en desarrollo activo. Puede estar sujeta a cambios frecuentes.

🛠️ En este momento estoy trabajando en la implementación de los tests no paramétricos. Ver [`feature/non-parametric`](https://github.com/eduvicgar/hzero/tree/feature/non-parametrical)


## ✨ Características principales

- Implementación de distribuciones usadas en contraste de hipótesis (como t de Student, chi-cuadrado, etc.)
- Soporte para contrastes paramétricos (media, varianza, diferencia de medias y cociente de varianzas)
- Visualización de resultados con **matplotlib**
- Uso extensivo de **NumPy** para operaciones numéricas eficientes

> [!WARNING]
> La mayoría de las clases proporcionadas no han sido meticulosamente testeadas, lo que puede dar lugar a resultados inesperados en casos extremos.
> A lo largo del desarrollo de la librería se corregirán aquellos fallos que puedan surgir. También puedes presentar un issue si te encuentras
> con un error.

## 📦 Tecnologías utilizadas

- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## 🔧 Instalación

> [!NOTE]
> Por el momento, no hay una forma estándar de instalar hzero. Para usarlo, puedes clonar este repositorio:

```bash
git clone https://github.com/eduvicgar/hzero.git
cd hzero
```

Luego puedes importar los módulos directamente desde tu entorno local de desarrollo.

## 📄 Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

## ✍️ Autor
eduvicgar

## 📌 Próximas mejoras
- Constrastes de hipótesis no paramétricos como chi cuadrado de Pearson y Kolgomorov-Smirnoff
- Más tests automatizados
- Más documentación y notebooks de ejemplo
